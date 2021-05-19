// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Physical query planner

use std::sync::Arc;

use super::{
    aggregates, empty::EmptyExec, expressions::binary, functions,
    hash_join::PartitionMode, udaf, union::UnionExec,
};
use crate::error::{DataFusionError, Result};
use crate::execution::context::ExecutionContextState;
use crate::logical_plan::{
    DFSchema, Expr, LogicalPlan, Operator, Partitioning as LogicalPartitioning, PlanType,
    StringifiedPlan, UserDefinedLogicalNode,
};
use crate::physical_plan::explain::ExplainExec;
use crate::physical_plan::expressions::{CaseExpr, Column, Literal, PhysicalSortExpr};
use crate::physical_plan::filter::FilterExec;
use crate::physical_plan::hash_aggregate::{
    AggregateMode, AggregateStrategy, HashAggregateExec,
};
use crate::physical_plan::hash_join::HashJoinExec;
use crate::physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use crate::physical_plan::merge::MergeExec;
use crate::physical_plan::merge_join::MergeJoinExec;
use crate::physical_plan::merge_sort::{MergeReSortExec, MergeSortExec};
use crate::physical_plan::projection::ProjectionExec;
use crate::physical_plan::repartition::RepartitionExec;
use crate::physical_plan::sort::SortExec;
use crate::physical_plan::udf;
use crate::physical_plan::{expressions, ColumnarValue};
use crate::physical_plan::{hash_utils, Partitioning};
use crate::physical_plan::{AggregateExpr, ExecutionPlan, PhysicalExpr, PhysicalPlanner};
use crate::prelude::JoinType;
use crate::scalar::ScalarValue;
use crate::variable::VarType;
use arrow::compute::can_cast_types;

use crate::cube_ext::join::CrossJoinPlanner;
use crate::cube_ext::joinagg::CrossJoinAggPlanner;
use crate::physical_plan::alias::AliasedSchemaExec;
use crate::physical_plan::skip::SkipExec;
use arrow::array::Int32Array;
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use expressions::col;
use itertools::Itertools;
use log::debug;

/// This trait exposes the ability to plan an [`ExecutionPlan`] out of a [`LogicalPlan`].
pub trait ExtensionPlanner {
    /// Create a physical plan for a [`UserDefinedLogicalNode`].
    /// This errors when the planner knows how to plan the concrete implementation of `node`
    /// but errors while doing so, and `None` when the planner does not know how to plan the `node`
    /// and wants to delegate the planning to another [`ExtensionPlanner`].
    fn plan_extension(
        &self,
        node: &dyn UserDefinedLogicalNode,
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx_state: &ExecutionContextState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>>;
}

/// Default single node physical query planner that converts a
/// `LogicalPlan` to an `ExecutionPlan` suitable for execution.
pub struct DefaultPhysicalPlanner {
    extension_planners: Vec<Arc<dyn ExtensionPlanner + Send + Sync>>,
}

impl Default for DefaultPhysicalPlanner {
    fn default() -> Self {
        Self {
            extension_planners: vec![
                Arc::new(CrossJoinPlanner {}),
                Arc::new(CrossJoinAggPlanner {}),
            ],
        }
    }
}

impl PhysicalPlanner for DefaultPhysicalPlanner {
    /// Create a physical plan from a logical plan
    fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        ctx_state: &ExecutionContextState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let plan = self.create_initial_plan(logical_plan, ctx_state)?;
        self.optimize_plan(plan, ctx_state)
    }
}

impl DefaultPhysicalPlanner {
    /// Create a physical planner that uses `extension_planners` to
    /// plan user-defined logical nodes [`LogicalPlan::Extension`].
    /// The planner uses the first [`ExtensionPlanner`] to return a non-`None`
    /// plan.
    pub fn with_extension_planners(
        mut extension_planners: Vec<Arc<dyn ExtensionPlanner + Send + Sync>>,
    ) -> Self {
        extension_planners.insert(0, Arc::new(CrossJoinPlanner {}));
        extension_planners.insert(1, Arc::new(CrossJoinAggPlanner {}));
        Self { extension_planners }
    }

    /// Optimize a physical plan
    fn optimize_plan(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        ctx_state: &ExecutionContextState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let optimizers = &ctx_state.config.physical_optimizers;
        debug!("Physical plan:\n{:?}", plan);

        let mut new_plan = plan;
        for optimizer in optimizers {
            new_plan = optimizer.optimize(new_plan, &ctx_state.config)?;
        }
        debug!("Optimized physical plan:\n{:?}", new_plan);
        Ok(new_plan)
    }

    /// Create a physical plan from a logical plan
    fn create_initial_plan(
        &self,
        logical_plan: &LogicalPlan,
        ctx_state: &ExecutionContextState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let batch_size = ctx_state.config.batch_size;

        match logical_plan {
            LogicalPlan::TableScan {
                source,
                projection,
                filters,
                alias,
                limit,
                ..
            } => Ok(AliasedSchemaExec::wrap(
                alias.clone(),
                source.scan(projection, batch_size, filters, *limit)?,
            )),
            LogicalPlan::Aggregate {
                input,
                group_expr,
                aggr_expr,
                ..
            } => {
                // Initially need to perform the aggregate and then merge the partitions
                let input_exec = self.create_initial_plan(input, ctx_state)?;
                let input_schema = input_exec.schema();
                let physical_input_schema = input_exec.as_ref().schema();
                let logical_input_schema = input.as_ref().schema();

                let groups = group_expr
                    .iter()
                    .map(|e| {
                        tuple_err((
                            self.create_physical_expr(
                                e,
                                &physical_input_schema,
                                ctx_state,
                            ),
                            e.name(&logical_input_schema),
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let aggregates = aggr_expr
                    .iter()
                    .map(|e| {
                        self.create_aggregate_expr(
                            e,
                            &logical_input_schema,
                            &physical_input_schema,
                            ctx_state,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;

                let strategy = compute_aggregation_strategy(input_exec.as_ref(), &groups);
                // TODO: fix cubestore planning and re-enable.
                if false && input_exec.output_partitioning().partition_count() == 1 {
                    // A single pass is enough for 1 partition.
                    return Ok(Arc::new(HashAggregateExec::try_new(
                        strategy,
                        AggregateMode::Full,
                        groups,
                        aggregates,
                        input_exec,
                        input_schema.clone(),
                    )?));
                }

                let mut initial_aggr: Arc<dyn ExecutionPlan> =
                    Arc::new(HashAggregateExec::try_new(
                        strategy,
                        AggregateMode::Partial,
                        groups.clone(),
                        aggregates.clone(),
                        input_exec,
                        input_schema.clone(),
                    )?);

                if strategy == AggregateStrategy::InplaceSorted
                    && initial_aggr.output_partitioning().partition_count() != 1
                    && !groups.is_empty()
                {
                    initial_aggr = Arc::new(MergeSortExec::try_new(
                        initial_aggr,
                        groups.iter().map(|(_, name)| name.clone()).collect(),
                    )?);
                }

                let final_group: Vec<Arc<dyn PhysicalExpr>> =
                    (0..groups.len()).map(|i| col(&groups[i].1)).collect();

                // construct a second aggregation, keeping the final column name equal to the first aggregation
                // and the expressions corresponding to the respective aggregate
                Ok(Arc::new(HashAggregateExec::try_new(
                    strategy,
                    AggregateMode::Final,
                    final_group
                        .iter()
                        .enumerate()
                        .map(|(i, expr)| (expr.clone(), groups[i].1.clone()))
                        .collect(),
                    aggregates,
                    initial_aggr,
                    input_schema,
                )?))
            }
            LogicalPlan::Projection { input, expr, .. } => {
                let input_exec = self.create_initial_plan(input, ctx_state)?;
                let input_schema = input.as_ref().schema();
                let runtime_expr = expr
                    .iter()
                    .map(|e| {
                        tuple_err((
                            self.create_physical_expr(
                                e,
                                &input_exec.schema(),
                                &ctx_state,
                            ),
                            e.name(&input_schema),
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Arc::new(ProjectionExec::try_new(runtime_expr, input_exec)?))
            }
            LogicalPlan::Filter {
                input, predicate, ..
            } => {
                let input = self.create_initial_plan(input, ctx_state)?;
                let input_schema = input.as_ref().schema();
                let runtime_expr =
                    self.create_physical_expr(predicate, &input_schema, ctx_state)?;
                Ok(Arc::new(FilterExec::try_new(runtime_expr, input)?))
            }
            LogicalPlan::Union { inputs, alias, .. } => {
                let physical_plans = inputs
                    .iter()
                    .map(|input| self.create_physical_plan(input, ctx_state))
                    .collect::<Result<Vec<_>>>()?;
                let sorted_on = physical_plans
                    .iter()
                    .map(|p| {
                        self.merge_sort_node(p.clone()).and_then(|n| {
                            n.as_any()
                                .downcast_ref::<MergeSortExec>()
                                .map(|n| n.columns.clone())
                        })
                    })
                    .unique()
                    .collect::<Vec<_>>();
                let merge_node: Arc<dyn ExecutionPlan> =
                    if sorted_on.iter().all(|on| on.is_some()) && sorted_on.len() == 1 {
                        Arc::new(MergeSortExec::try_new(
                            Arc::new(UnionExec::new(physical_plans)),
                            sorted_on[0].as_ref().unwrap().clone(),
                        )?)
                    } else {
                        Arc::new(MergeExec::new(Arc::new(UnionExec::new(physical_plans))))
                    };
                Ok(AliasedSchemaExec::wrap(alias.clone(), merge_node))
            }
            LogicalPlan::Repartition {
                input,
                partitioning_scheme,
            } => {
                let input = self.create_initial_plan(input, ctx_state)?;
                let input_schema = input.schema();
                let physical_partitioning = match partitioning_scheme {
                    LogicalPartitioning::RoundRobinBatch(n) => {
                        Partitioning::RoundRobinBatch(*n)
                    }
                    LogicalPartitioning::Hash(expr, n) => {
                        let runtime_expr = expr
                            .iter()
                            .map(|e| {
                                self.create_physical_expr(e, &input_schema, &ctx_state)
                            })
                            .collect::<Result<Vec<_>>>()?;
                        Partitioning::Hash(runtime_expr, *n)
                    }
                };
                Ok(Arc::new(RepartitionExec::try_new(
                    input,
                    physical_partitioning,
                )?))
            }
            LogicalPlan::Sort { expr, input, .. } => {
                let input = self.create_initial_plan(input, ctx_state)?;
                let input_schema = input.as_ref().schema();

                let sort_expr = expr
                    .iter()
                    .map(|e| match e {
                        Expr::Sort {
                            expr,
                            asc,
                            nulls_first,
                        } => self.create_physical_sort_expr(
                            expr,
                            &input_schema,
                            SortOptions {
                                descending: !*asc,
                                nulls_first: *nulls_first,
                            },
                            ctx_state,
                        ),
                        _ => Err(DataFusionError::Plan(
                            "Sort only accepts sort expressions".to_string(),
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;

                Ok(Arc::new(SortExec::try_new(sort_expr, input)?))
            }
            LogicalPlan::Join {
                left,
                right,
                on: keys,
                join_type,
                ..
            } => {
                let left = self.create_initial_plan(left, ctx_state)?;
                let right = self.create_initial_plan(right, ctx_state)?;
                let physical_join_type = match join_type {
                    JoinType::Inner => hash_utils::JoinType::Inner,
                    JoinType::Left => hash_utils::JoinType::Left,
                    JoinType::Right => hash_utils::JoinType::Right,
                };
                let left_schema = left.schema();
                let right_schema = right.schema();
                let keys = keys
                    .iter()
                    .map(|(l, r)| -> Result<_> {
                        Ok((
                            left_schema.lookup_field_by_string_name(l)?.qualified_name(),
                            right_schema
                                .lookup_field_by_string_name(r)?
                                .qualified_name(),
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                if let (Some(left_node), Some(right_node)) = (
                    self.merge_sort_node(left.clone()),
                    self.merge_sort_node(right.clone()),
                ) {
                    let left_to_join =
                        if left_node.as_any().downcast_ref::<MergeJoinExec>().is_some() {
                            Arc::new(MergeReSortExec::try_new(
                                left,
                                keys.iter().map(|(l, _)| l.to_string()).collect(),
                            )?)
                        } else {
                            left
                        };

                    let right_to_join = if right_node
                        .as_any()
                        .downcast_ref::<MergeJoinExec>()
                        .is_some()
                    {
                        Arc::new(MergeReSortExec::try_new(
                            right,
                            keys.iter().map(|(_, r)| r.to_string()).collect(),
                        )?)
                    } else {
                        right
                    };
                    Ok(Arc::new(MergeJoinExec::try_new(
                        left_to_join,
                        right_to_join,
                        &keys,
                        &physical_join_type,
                    )?))
                } else {
                    if ctx_state.config.concurrency > 1
                        && ctx_state.config.repartition_joins
                    {
                        let left_expr = keys.iter().map(|x| col(&x.0)).collect();
                        let right_expr = keys.iter().map(|x| col(&x.1)).collect();

                        // Use hash partition by defualt to parallelize hash joins
                        Ok(Arc::new(HashJoinExec::try_new(
                            Arc::new(RepartitionExec::try_new(
                                left,
                                Partitioning::Hash(
                                    left_expr,
                                    ctx_state.config.concurrency,
                                ),
                            )?),
                            Arc::new(RepartitionExec::try_new(
                                right,
                                Partitioning::Hash(
                                    right_expr,
                                    ctx_state.config.concurrency,
                                ),
                            )?),
                            &keys,
                            &physical_join_type,
                            PartitionMode::Partitioned,
                        )?))
                    } else {
                        Ok(Arc::new(HashJoinExec::try_new(
                            left,
                            right,
                            &keys,
                            &physical_join_type,
                            PartitionMode::CollectLeft,
                        )?))
                    }
                }
            }
            LogicalPlan::EmptyRelation {
                produce_one_row,
                schema,
            } => Ok(Arc::new(EmptyExec::new(
                *produce_one_row,
                SchemaRef::new(schema.as_ref().to_owned().into()),
            ))),
            LogicalPlan::Limit { input, n, .. } => {
                let limit = *n;
                let input = self.create_initial_plan(input, ctx_state)?;

                // GlobalLimitExec requires a single partition for input
                let input = if input.output_partitioning().partition_count() == 1 {
                    input
                } else {
                    // Apply a LocalLimitExec to each partition. The optimizer will also insert
                    // a MergeExec between the GlobalLimitExec and LocalLimitExec
                    Arc::new(LocalLimitExec::new(input, limit))
                };

                Ok(Arc::new(GlobalLimitExec::new(input, limit)))
            }
            LogicalPlan::Skip { input, n, .. } => {
                let skip = *n;
                let input = self.create_physical_plan(input, ctx_state)?;

                Ok(Arc::new(SkipExec::new(input, skip)))
            }
            LogicalPlan::CreateExternalTable { .. } => {
                // There is no default plan for "CREATE EXTERNAL
                // TABLE" -- it must be handled at a higher level (so
                // that the appropriate table can be registered with
                // the context)
                Err(DataFusionError::Internal(
                    "Unsupported logical plan: CreateExternalTable".to_string(),
                ))
            }
            LogicalPlan::Explain {
                verbose,
                plan,
                stringified_plans,
                schema,
            } => {
                let input = self.create_initial_plan(plan, ctx_state)?;

                let mut stringified_plans = stringified_plans
                    .iter()
                    .filter(|s| s.should_display(*verbose))
                    .cloned()
                    .collect::<Vec<_>>();

                // add in the physical plan if requested
                if *verbose {
                    stringified_plans.push(StringifiedPlan::new(
                        PlanType::PhysicalPlan,
                        format!("{:#?}", input),
                    ));
                }
                Ok(Arc::new(ExplainExec::new(
                    SchemaRef::new(schema.as_ref().to_owned().into()),
                    stringified_plans,
                )))
            }
            LogicalPlan::Extension { node } => {
                let inputs = node
                    .inputs()
                    .into_iter()
                    .map(|input_plan| self.create_initial_plan(input_plan, ctx_state))
                    .collect::<Result<Vec<_>>>()?;

                let maybe_plan = self.extension_planners.iter().try_fold(
                    None,
                    |maybe_plan, planner| {
                        if let Some(plan) = maybe_plan {
                            Ok(Some(plan))
                        } else {
                            planner.plan_extension(node.as_ref(), &inputs, ctx_state)
                        }
                    },
                )?;
                let plan = maybe_plan.ok_or_else(|| DataFusionError::Plan(format!(
                    "No installed planner was able to convert the custom node to an execution plan: {:?}", node
                )))?;

                // Ensure the ExecutionPlan's  schema matches the
                // declared logical schema to catch and warn about
                // logic errors when creating user defined plans.
                if !Self::logical_schema_matches_physical(node.schema(), &plan.schema()) {
                    Err(DataFusionError::Plan(format!(
                        "Extension planner for {:?} created an ExecutionPlan with mismatched schema. \
                         LogicalPlan schema: {:?}, ExecutionPlan schema: {:?}",
                        node, node.schema(), plan.schema()
                    )))
                } else {
                    Ok(plan)
                }
            }
        }
    }

    fn logical_schema_matches_physical(logical: &DFSchema, physical: &DFSchema) -> bool {
        let logical = logical.fields();
        let physical = physical.fields();
        if logical.len() != physical.len() {
            return false;
        }

        for i in 0..logical.len() {
            if logical[i].data_type() != physical[i].data_type()
                || logical[i].qualifier() != physical[i].qualifier()
                || logical[i].name() != physical[i].name()
            {
                return false;
            }
            // We do not check for mismatch in nullable():
            //  - binary expressions can become nullable because of the added TryCast nodes,
            //  - constant evaluation can turn nullable expressions to non-nullable.
        }
        return true;
    }

    fn merge_sort_node(
        &self,
        node: Arc<dyn ExecutionPlan>,
    ) -> Option<Arc<dyn ExecutionPlan>> {
        if node.as_any().downcast_ref::<MergeSortExec>().is_some()
            || node.as_any().downcast_ref::<MergeJoinExec>().is_some()
        {
            Some(node.clone())
        } else if let Some(aliased) = node.as_any().downcast_ref::<AliasedSchemaExec>() {
            self.merge_sort_node(aliased.children()[0].clone())
        } else if let Some(aliased) = node.as_any().downcast_ref::<FilterExec>() {
            self.merge_sort_node(aliased.children()[0].clone())
        } else if let Some(aliased) = node.as_any().downcast_ref::<ProjectionExec>() {
            // TODO
            self.merge_sort_node(aliased.children()[0].clone())
        } else {
            None
        }
    }

    /// Create a physical expression from a logical expression
    pub fn create_physical_expr(
        &self,
        e: &Expr,
        input_schema: &DFSchema,
        ctx_state: &ExecutionContextState,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        match e {
            Expr::Alias(expr, ..) => {
                Ok(self.create_physical_expr(expr, input_schema, ctx_state)?)
            }
            Expr::Column(name, relation) => {
                // check that name exists
                input_schema
                    .field_with_name(relation.as_ref().map(|r| r.as_str()), &name)?;
                Ok(Arc::new(Column::new_with_alias(name, relation.clone())))
            }
            Expr::Literal(value) => Ok(Arc::new(Literal::new(value.clone()))),
            Expr::ScalarVariable(variable_names) => {
                if &variable_names[0][0..2] == "@@" {
                    match ctx_state.var_provider.get(&VarType::System) {
                        Some(provider) => {
                            let scalar_value =
                                provider.get_value(variable_names.clone())?;
                            Ok(Arc::new(Literal::new(scalar_value)))
                        }
                        _ => Err(DataFusionError::Plan(
                            "No system variable provider found".to_string(),
                        )),
                    }
                } else {
                    match ctx_state.var_provider.get(&VarType::UserDefined) {
                        Some(provider) => {
                            let scalar_value =
                                provider.get_value(variable_names.clone())?;
                            Ok(Arc::new(Literal::new(scalar_value)))
                        }
                        _ => Err(DataFusionError::Plan(
                            "No user defined variable provider found".to_string(),
                        )),
                    }
                }
            }
            Expr::BinaryExpr { left, op, right } => {
                let lhs = self.create_physical_expr(left, input_schema, ctx_state)?;
                let rhs = self.create_physical_expr(right, input_schema, ctx_state)?;
                self.evaluate_constants(
                    binary(lhs.clone(), *op, rhs.clone(), input_schema)?,
                    vec![lhs, rhs],
                )
            }
            Expr::Case {
                expr,
                when_then_expr,
                else_expr,
                ..
            } => {
                let expr: Option<Arc<dyn PhysicalExpr>> = if let Some(e) = expr {
                    Some(self.create_physical_expr(
                        e.as_ref(),
                        input_schema,
                        ctx_state,
                    )?)
                } else {
                    None
                };
                let when_expr = when_then_expr
                    .iter()
                    .map(|(w, _)| {
                        self.create_physical_expr(w.as_ref(), input_schema, ctx_state)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let then_expr = when_then_expr
                    .iter()
                    .map(|(_, t)| {
                        self.create_physical_expr(t.as_ref(), input_schema, ctx_state)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let when_then_expr: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> =
                    when_expr
                        .iter()
                        .zip(then_expr.iter())
                        .map(|(w, t)| (w.clone(), t.clone()))
                        .collect();
                let else_expr: Option<Arc<dyn PhysicalExpr>> = if let Some(e) = else_expr
                {
                    Some(self.create_physical_expr(
                        e.as_ref(),
                        input_schema,
                        ctx_state,
                    )?)
                } else {
                    None
                };
                let args = when_expr
                    .iter()
                    .chain(then_expr.iter())
                    .chain(else_expr.iter())
                    .chain(expr.iter())
                    .cloned()
                    .collect();
                let case_expr =
                    Arc::new(CaseExpr::try_new(expr, &when_then_expr, else_expr)?);
                self.evaluate_constants(case_expr, args)
            }
            Expr::Cast { expr, data_type } => {
                let input = self.create_physical_expr(expr, input_schema, ctx_state)?;
                self.evaluate_constants(
                    expressions::cast(input.clone(), input_schema, data_type.clone())?,
                    vec![input],
                )
            }
            Expr::TryCast { expr, data_type } => {
                let input = self.create_physical_expr(expr, input_schema, ctx_state)?;
                self.evaluate_constants(
                    expressions::try_cast(
                        input.clone(),
                        input_schema,
                        data_type.clone(),
                    )?,
                    vec![input],
                )
            }
            Expr::Not(expr) => {
                let input = self.create_physical_expr(expr, input_schema, ctx_state)?;
                self.evaluate_constants(
                    expressions::not(input.clone(), input_schema)?,
                    vec![input],
                )
            }
            Expr::Negative(expr) => {
                let input = self.create_physical_expr(expr, input_schema, ctx_state)?;
                self.evaluate_constants(
                    expressions::negative(input.clone(), input_schema)?,
                    vec![input],
                )
            }
            Expr::IsNull(expr) => {
                let input = self.create_physical_expr(expr, input_schema, ctx_state)?;
                self.evaluate_constants(expressions::is_null(input.clone())?, vec![input])
            }
            Expr::IsNotNull(expr) => {
                let input = self.create_physical_expr(expr, input_schema, ctx_state)?;
                self.evaluate_constants(
                    expressions::is_not_null(input.clone())?,
                    vec![input],
                )
            }
            Expr::ScalarFunction { fun, args } => {
                let physical_args = args
                    .iter()
                    .map(|e| self.create_physical_expr(e, input_schema, ctx_state))
                    .collect::<Result<Vec<_>>>()?;
                self.evaluate_constants(
                    functions::create_physical_expr(fun, &physical_args, input_schema)?,
                    physical_args,
                )
            }
            Expr::ScalarUDF { fun, args } => {
                let mut physical_args = vec![];
                for e in args {
                    physical_args.push(self.create_physical_expr(
                        e,
                        input_schema,
                        ctx_state,
                    )?);
                }

                self.evaluate_constants(
                    udf::create_physical_expr(
                        fun.clone().as_ref(),
                        &physical_args,
                        input_schema,
                    )?,
                    physical_args,
                )
            }
            Expr::Between {
                expr,
                negated,
                low,
                high,
            } => {
                let value_expr =
                    self.create_physical_expr(expr, input_schema, ctx_state)?;
                let low_expr = self.create_physical_expr(low, input_schema, ctx_state)?;
                let high_expr =
                    self.create_physical_expr(high, input_schema, ctx_state)?;

                // rewrite the between into the two binary operators
                let binary_expr = binary(
                    binary(value_expr.clone(), Operator::GtEq, low_expr, input_schema)?,
                    Operator::And,
                    binary(value_expr.clone(), Operator::LtEq, high_expr, input_schema)?,
                    input_schema,
                );

                if *negated {
                    expressions::not(binary_expr?, input_schema)
                } else {
                    binary_expr
                }
            }
            Expr::InList {
                expr,
                list,
                negated,
            } => match expr.as_ref() {
                Expr::Literal(ScalarValue::Utf8(None)) => {
                    Ok(expressions::lit(ScalarValue::Boolean(None)))
                }
                _ => {
                    let value_expr =
                        self.create_physical_expr(expr, input_schema, ctx_state)?;
                    let value_expr_data_type = value_expr.data_type(input_schema)?;

                    let list_exprs =
                        list.iter()
                            .map(|expr| match expr {
                                Expr::Literal(ScalarValue::Utf8(None)) => self
                                    .create_physical_expr(expr, input_schema, ctx_state),
                                _ => {
                                    let list_expr = self.create_physical_expr(
                                        expr,
                                        input_schema,
                                        ctx_state,
                                    )?;
                                    let list_expr_data_type =
                                        list_expr.data_type(input_schema)?;

                                    if list_expr_data_type == value_expr_data_type {
                                        Ok(list_expr)
                                    } else if can_cast_types(
                                        &list_expr_data_type,
                                        &value_expr_data_type,
                                    ) {
                                        expressions::cast(
                                            list_expr,
                                            input_schema,
                                            value_expr.data_type(input_schema)?,
                                        )
                                    } else {
                                        Err(DataFusionError::Plan(format!(
                                            "Unsupported CAST from {:?} to {:?}",
                                            list_expr_data_type, value_expr_data_type
                                        )))
                                    }
                                }
                            })
                            .collect::<Result<Vec<_>>>()?;

                    expressions::in_list(value_expr, list_exprs, negated)
                }
            },
            other => Err(DataFusionError::NotImplemented(format!(
                "Physical plan does not support logical expression {:?}",
                other
            ))),
        }
    }

    fn evaluate_constants(
        &self,
        res_expr: Arc<dyn PhysicalExpr>,
        inputs: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        if inputs
            .iter()
            .all(|i| i.as_any().downcast_ref::<Literal>().is_some())
        {
            Ok(evaluate_const(res_expr)?)
        } else {
            Ok(res_expr)
        }
    }

    /// Create an aggregate expression from a logical expression
    pub fn create_aggregate_expr(
        &self,
        e: &Expr,
        logical_input_schema: &DFSchema,
        physical_input_schema: &DFSchema,
        ctx_state: &ExecutionContextState,
    ) -> Result<Arc<dyn AggregateExpr>> {
        // unpack aliased logical expressions, e.g. "sum(col) as total"
        let (name, e) = match e {
            Expr::Alias(sub_expr, alias) => (alias.clone(), sub_expr.as_ref()),
            _ => (e.name(logical_input_schema)?, e),
        };

        match e {
            Expr::AggregateFunction {
                fun,
                distinct,
                args,
                ..
            } => {
                let args = args
                    .iter()
                    .map(|e| {
                        self.create_physical_expr(e, physical_input_schema, ctx_state)
                    })
                    .collect::<Result<Vec<_>>>()?;
                aggregates::create_aggregate_expr(
                    fun,
                    *distinct,
                    &args,
                    physical_input_schema,
                    name,
                )
            }
            Expr::AggregateUDF { fun, args, .. } => {
                let args = args
                    .iter()
                    .map(|e| {
                        self.create_physical_expr(e, physical_input_schema, ctx_state)
                    })
                    .collect::<Result<Vec<_>>>()?;

                udaf::create_aggregate_expr(fun, &args, physical_input_schema, name)
            }
            other => Err(DataFusionError::Internal(format!(
                "Invalid aggregate expression '{:?}'",
                other
            ))),
        }
    }

    /// Create an aggregate expression from a logical expression
    pub fn create_physical_sort_expr(
        &self,
        e: &Expr,
        input_schema: &DFSchema,
        options: SortOptions,
        ctx_state: &ExecutionContextState,
    ) -> Result<PhysicalSortExpr> {
        Ok(PhysicalSortExpr {
            expr: self.create_physical_expr(e, input_schema, ctx_state)?,
            options,
        })
    }
}

/// Evaluate PhysicalExpr for a single row dummy batch
pub fn evaluate_const(expr: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
    // This is a dummy array. Consider using special batch implementation?
    let array = Int32Array::from(vec![1]);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, true)])),
        vec![Arc::new(array)],
    )?;
    let value = expr.evaluate(&batch)?;
    let scalar = match value {
        ColumnarValue::Scalar(value) => value,
        ColumnarValue::Array(a) => ScalarValue::try_from_array(&a, 0)?,
    };
    Ok(Arc::new(Literal::new(scalar)))
}

/// Returns the most efficient aggregation strategy for the given input.
pub fn compute_aggregation_strategy(
    input: &dyn ExecutionPlan,
    group_key: &[(Arc<dyn PhysicalExpr>, String)],
) -> AggregateStrategy {
    if !group_key.is_empty() && input_sorted_by_group_key(input, &group_key) {
        AggregateStrategy::InplaceSorted
    } else {
        AggregateStrategy::Hash
    }
}

fn input_sorted_by_group_key(
    input: &dyn ExecutionPlan,
    group_key: &[(Arc<dyn PhysicalExpr>, String)],
) -> bool {
    assert!(!group_key.is_empty());
    let hints = input.output_hints();
    // We check the group key is a prefix of the sort key.
    let sort_key = hints.sort_order;
    if sort_key.is_none() {
        return false;
    }
    let sort_key = sort_key.unwrap();
    // Tracks which elements of sort key are used in the group key or have a single value.
    let mut sort_key_hit = vec![false; sort_key.len()];
    for (g, _) in group_key {
        let col = g.as_any().downcast_ref::<Column>();
        if col.is_none() {
            return false;
        }
        let input_col = input.schema().index_of(col.unwrap().name());
        if input_col.is_err() {
            return false;
        }
        let input_col = input_col.unwrap();
        let sort_key_pos = sort_key.iter().find_position(|i| **i == input_col);
        if sort_key_pos.is_none() {
            return false;
        }
        sort_key_hit[sort_key_pos.unwrap().0] = true;
    }
    for i in 0..sort_key.len() {
        if hints.single_value_columns.contains(&sort_key[i]) {
            sort_key_hit[i] = true;
        }
    }

    // At this point all elements of the group key mapped into some column of the sort key.
    // This checks the group key is mapped into a prefix of the sort key.
    sort_key_hit
        .iter()
        .skip_while(|present| **present)
        .skip_while(|present| !**present)
        .next()
        .is_none()
}

fn tuple_err<T, R>(value: (Result<T>, Result<R>)) -> Result<(T, R)> {
    match value {
        (Ok(e), Ok(e1)) => Ok((e, e1)),
        (Err(e), Ok(_)) => Err(e),
        (Ok(_), Err(e1)) => Err(e1),
        (Err(e), Err(_)) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical_plan::{csv::CsvReadOptions, expressions, Partitioning};
    use crate::prelude::ExecutionConfig;
    use crate::scalar::ScalarValue;
    use crate::{
        catalog::catalog::MemoryCatalogList,
        logical_plan::{DFField, DFSchema, DFSchemaRef, ToDFSchema},
    };
    use crate::{
        logical_plan::{col, lit, sum, LogicalPlanBuilder},
        physical_plan::SendableRecordBatchStream,
    };
    use arrow::datatypes::{DataType, Field, Schema};
    use async_trait::async_trait;
    use fmt::Debug;
    use std::{any::Any, collections::HashMap, fmt};

    fn make_ctx_state() -> ExecutionContextState {
        ExecutionContextState {
            catalog_list: Arc::new(MemoryCatalogList::new()),
            scalar_functions: HashMap::new(),
            var_provider: HashMap::new(),
            aggregate_functions: HashMap::new(),
            config: ExecutionConfig::new(),
        }
    }

    fn plan(logical_plan: &LogicalPlan) -> Result<Arc<dyn ExecutionPlan>> {
        let ctx_state = make_ctx_state();
        let planner = DefaultPhysicalPlanner::default();
        planner.create_physical_plan(logical_plan, &ctx_state)
    }

    #[test]
    fn test_all_operators() -> Result<()> {
        let testdata = arrow::util::test_util::arrow_test_data();
        let path = format!("{}/csv/aggregate_test_100.csv", testdata);

        let options = CsvReadOptions::new().schema_infer_max_records(100);
        let logical_plan = LogicalPlanBuilder::scan_csv(&path, options, None)?
            // filter clause needs the type coercion rule applied
            .filter(col("c7").lt(lit(5_u8)))?
            .project(vec![col("c1"), col("c2")])?
            .aggregate(vec![col("c1")], vec![sum(col("c2"))])?
            .sort(vec![col("c1").sort(true, true)])?
            .limit(10)?
            .build()?;

        let plan = plan(&logical_plan)?;

        // verify that the plan correctly casts u8 to i64
        // the cast here is implicit so has CastOptions with safe=true
        let expected = "BinaryExpr { left: Column { name: \"c7\", relation: None }, op: Lt, right: TryCastExpr { expr: Literal { value: UInt8(5) }, cast_type: Int64 } }";
        assert!(format!("{:?}", plan).contains(expected));

        Ok(())
    }

    #[test]
    fn test_create_not() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Boolean, true)]);

        let planner = DefaultPhysicalPlanner::default();

        let expr = planner.create_physical_expr(
            &col("a").not(),
            &schema.clone().to_dfschema()?,
            &make_ctx_state(),
        )?;
        let expected = expressions::not(expressions::col("a"), &schema.to_dfschema()?)?;

        assert_eq!(format!("{:?}", expr), format!("{:?}", expected));

        Ok(())
    }

    #[test]
    fn test_with_csv_plan() -> Result<()> {
        let testdata = arrow::util::test_util::arrow_test_data();
        let path = format!("{}/csv/aggregate_test_100.csv", testdata);

        let options = CsvReadOptions::new().schema_infer_max_records(100);
        let logical_plan = LogicalPlanBuilder::scan_csv(&path, options, None)?
            .filter(col("c7").lt(col("c12")))?
            .build()?;

        let plan = plan(&logical_plan)?;

        // c12 is f64, c7 is u8 -> cast c7 to f64
        // the cast here is implicit so has CastOptions with safe=true
        let expected = "predicate: BinaryExpr { left: TryCastExpr { expr: Column { name: \"c7\", relation: None }, cast_type: Float64 }, op: Lt, right: Column { name: \"c12\", relation: None } }";
        assert!(format!("{:?}", plan).contains(expected));
        Ok(())
    }

    #[test]
    #[ignore = "Cube Store coerces strings to numerics"]
    fn errors() -> Result<()> {
        let testdata = arrow::util::test_util::arrow_test_data();
        let path = format!("{}/csv/aggregate_test_100.csv", testdata);
        let options = CsvReadOptions::new().schema_infer_max_records(100);

        let bool_expr = col("c1").eq(col("c1"));
        let cases = vec![
            // utf8 < u32
            col("c1").lt(col("c2")),
            // utf8 AND utf8
            col("c1").and(col("c1")),
            // u8 AND u8
            col("c3").and(col("c3")),
            // utf8 = u32
            col("c1").eq(col("c2")),
            // utf8 = bool
            col("c1").eq(bool_expr.clone()),
            // u32 AND bool
            col("c2").and(bool_expr),
            // utf8 LIKE u32
            col("c1").like(col("c2")),
        ];
        for case in cases {
            let logical_plan = LogicalPlanBuilder::scan_csv(&path, options, None)?
                .project(vec![case.clone()]);
            let message = format!(
                "Expression {:?} expected to error due to impossible coercion",
                case
            );
            assert!(logical_plan.is_err(), "{}", message);
        }
        Ok(())
    }

    #[test]
    fn default_extension_planner() {
        let ctx_state = make_ctx_state();
        let planner = DefaultPhysicalPlanner::default();
        let logical_plan = LogicalPlan::Extension {
            node: Arc::new(NoOpExtensionNode::default()),
        };
        let plan = planner.create_physical_plan(&logical_plan, &ctx_state);

        let expected_error =
            "No installed planner was able to convert the custom node to an execution plan: NoOp";
        match plan {
            Ok(_) => panic!("Expected planning failure"),
            Err(e) => assert!(
                e.to_string().contains(expected_error),
                "Error '{}' did not contain expected error '{}'",
                e.to_string(),
                expected_error
            ),
        }
    }

    #[test]
    fn bad_extension_planner() {
        // Test that creating an execution plan whose schema doesn't
        // match the logical plan's schema generates an error.
        let ctx_state = make_ctx_state();
        let planner = DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(
            BadExtensionPlanner {},
        )]);

        let logical_plan = LogicalPlan::Extension {
            node: Arc::new(NoOpExtensionNode::default()),
        };
        let plan = planner.create_physical_plan(&logical_plan, &ctx_state);

        let expected_error: &str = "Error during planning: \
        Extension planner for NoOp created an ExecutionPlan with mismatched schema. \
        LogicalPlan schema: DFSchema { fields: [\
            DFField { qualifier: None, field: Field { \
                name: \"a\", \
                data_type: Int32, \
                nullable: false, \
                dict_id: 0, \
                dict_is_ordered: false, \
                metadata: None } }\
        ] }, \
        ExecutionPlan schema: DFSchema { fields: [\
            DFField { qualifier: None, field: Field { \
                name: \"b\", \
                data_type: Int32, \
                nullable: false, \
                dict_id: 0, \
                dict_is_ordered: false, \
                metadata: None } }\
        ] }";
        match plan {
            Ok(_) => panic!("Expected planning failure"),
            Err(e) => assert!(
                e.to_string().contains(expected_error),
                "Error '{}' did not contain expected error '{}'",
                e.to_string(),
                expected_error
            ),
        }
    }

    #[test]
    #[ignore]
    fn in_list_types() -> Result<()> {
        let testdata = arrow::util::test_util::arrow_test_data();
        let path = format!("{}/csv/aggregate_test_100.csv", testdata);
        let options = CsvReadOptions::new().schema_infer_max_records(100);

        // expression: "a in ('a', 1)"
        let list = vec![
            Expr::Literal(ScalarValue::Utf8(Some("a".to_string()))),
            Expr::Literal(ScalarValue::Int64(Some(1))),
        ];
        let logical_plan = LogicalPlanBuilder::scan_csv(&path, options, None)?
            // filter clause needs the type coercion rule applied
            .filter(col("c12").lt(lit(0.05)))?
            .project(vec![col("c1").in_list(list, false)])?
            .build()?;
        let execution_plan = plan(&logical_plan)?;
        // verify that the plan correctly adds cast from Int64(1) to Utf8
        let expected = "InListExpr { expr: Column { name: \"c1\" }, list: [Literal { value: Utf8(\"a\") }, CastExpr { expr: Literal { value: Int64(1) }, cast_type: Utf8, cast_options: CastOptions { safe: false } }], negated: false }";
        println!("{:?}", execution_plan);
        assert!(format!("{:?}", execution_plan).contains(expected));

        // expression: "a in (true, 'a')"
        let list = vec![
            Expr::Literal(ScalarValue::Boolean(Some(true))),
            Expr::Literal(ScalarValue::Utf8(Some("a".to_string()))),
        ];
        let logical_plan = LogicalPlanBuilder::scan_csv(&path, options, None)?
            // filter clause needs the type coercion rule applied
            .filter(col("c12").lt(lit(0.05)))?
            .project(vec![col("c12").lt_eq(lit(0.025)).in_list(list, false)])?
            .build()?;
        let execution_plan = plan(&logical_plan);

        let expected_error = "Unsupported CAST from Utf8 to Boolean";
        match execution_plan {
            Ok(_) => panic!("Expected planning failure"),
            Err(e) => assert!(
                e.to_string().contains(expected_error),
                "Error '{}' did not contain expected error '{}'",
                e.to_string(),
                expected_error
            ),
        }

        Ok(())
    }

    #[test]
    fn hash_agg_input_schema() -> Result<()> {
        let testdata = arrow::util::test_util::arrow_test_data();
        let path = format!("{}/csv/aggregate_test_100.csv", testdata);

        let options = CsvReadOptions::new().schema_infer_max_records(100);
        let logical_plan = LogicalPlanBuilder::scan_csv(&path, options, None)?
            .aggregate(vec![col("c1")], vec![sum(col("c2"))])?
            .build()?;

        let execution_plan = plan(&logical_plan)?;
        let final_hash_agg = execution_plan
            .as_any()
            .downcast_ref::<HashAggregateExec>()
            .expect("hash aggregate");
        assert_eq!("SUM(c2)", final_hash_agg.schema().field(1).name());
        // we need access to the input to the partial aggregate so that other projects can
        // implement serde
        assert_eq!("c2", final_hash_agg.input_schema().field(1).name());

        Ok(())
    }

    /// An example extension node that doesn't do anything
    struct NoOpExtensionNode {
        schema: DFSchemaRef,
    }

    impl Default for NoOpExtensionNode {
        fn default() -> Self {
            Self {
                schema: DFSchemaRef::new(
                    DFSchema::new(vec![DFField::new(None, "a", DataType::Int32, false)])
                        .unwrap(),
                ),
            }
        }
    }

    impl Debug for NoOpExtensionNode {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "NoOp")
        }
    }

    impl UserDefinedLogicalNode for NoOpExtensionNode {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn inputs(&self) -> Vec<&LogicalPlan> {
            vec![]
        }

        fn schema(&self) -> &DFSchemaRef {
            &self.schema
        }

        fn expressions(&self) -> Vec<Expr> {
            vec![]
        }

        fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "NoOp")
        }

        fn from_template(
            &self,
            _exprs: &[Expr],
            _inputs: &[LogicalPlan],
        ) -> Arc<dyn UserDefinedLogicalNode + Send + Sync> {
            unimplemented!("NoOp");
        }
    }

    #[derive(Debug)]
    struct NoOpExecutionPlan {
        schema: DFSchemaRef,
    }

    #[async_trait]
    impl ExecutionPlan for NoOpExecutionPlan {
        /// Return a reference to Any that can be used for downcasting
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn schema(&self) -> DFSchemaRef {
            self.schema.clone()
        }

        fn output_partitioning(&self) -> Partitioning {
            Partitioning::UnknownPartitioning(1)
        }

        fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
            vec![]
        }

        fn with_new_children(
            &self,
            _children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            unimplemented!("NoOpExecutionPlan::with_new_children");
        }

        async fn execute(&self, _partition: usize) -> Result<SendableRecordBatchStream> {
            unimplemented!("NoOpExecutionPlan::execute");
        }
    }

    //  Produces an execution plan where the schema is mismatched from
    //  the logical plan node.
    struct BadExtensionPlanner {}

    impl ExtensionPlanner for BadExtensionPlanner {
        /// Create a physical plan for an extension node
        fn plan_extension(
            &self,
            _node: &dyn UserDefinedLogicalNode,
            _inputs: &[Arc<dyn ExecutionPlan>],
            _ctx_state: &ExecutionContextState,
        ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
            Ok(Some(Arc::new(NoOpExecutionPlan {
                schema: DFSchemaRef::new(
                    DFSchema::new(vec![DFField::new(None, "b", DataType::Int32, false)])
                        .unwrap(),
                ),
            })))
        }
    }
}

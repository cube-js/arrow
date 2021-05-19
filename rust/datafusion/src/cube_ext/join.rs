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

//! Join that works on arbitrary conditions.

use crate::cube_ext::stream::StreamWithSchema;
use crate::error::{DataFusionError, Result};
use crate::execution::context::ExecutionContextState;
use crate::logical_plan::{
    DFSchemaRef, Expr, LogicalPlan, PlanVisitor, UserDefinedLogicalNode,
};
use crate::physical_plan::coalesce_batches::concat_batches;
use crate::physical_plan::hash_utils::{build_join_schema, JoinType};
use crate::physical_plan::planner::{DefaultPhysicalPlanner, ExtensionPlanner};
use crate::physical_plan::{
    collect, ExecutionPlan, Partitioning, PhysicalExpr, SendableRecordBatchStream,
};
use arrow::array::{BooleanArray, UInt64Array};
use arrow::compute::{filter_record_batch, take};
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use futures::StreamExt;
use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Cross-join that supports complex conditions.
#[derive(Clone, Debug)]
pub struct CrossJoin {
    pub schema: DFSchemaRef,
    pub left: LogicalPlan,
    pub right: LogicalPlan,
    pub on: Expr,
}

impl CrossJoin {
    pub fn from_template_typed(
        &self,
        exprs: &[Expr],
        inputs: &[LogicalPlan],
    ) -> CrossJoin {
        assert_eq!(exprs.len(), 1);
        assert_eq!(inputs.len(), 2);
        CrossJoin {
            left: inputs[0].clone(),
            right: inputs[1].clone(),
            on: exprs[0].clone(),
            schema: self.schema.clone(),
        }
    }
}

impl UserDefinedLogicalNode for CrossJoin {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.left, &self.right]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![self.on.clone()]
    }

    fn fmt_for_explain(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "CrossJoin: {:?}", self.on)
    }

    fn from_template(
        &self,
        exprs: &[Expr],
        inputs: &[LogicalPlan],
    ) -> Arc<dyn UserDefinedLogicalNode + Send + Sync> {
        Arc::new(self.from_template_typed(exprs, inputs))
    }
}

pub fn contains_table_scan(p: &LogicalPlan) -> bool {
    struct Visitor {
        seen_scan: bool,
    }
    impl PlanVisitor for Visitor {
        type Error = ();

        fn pre_visit(
            &mut self,
            plan: &LogicalPlan,
        ) -> std::result::Result<bool, Self::Error> {
            match plan {
                LogicalPlan::TableScan { .. } => {
                    self.seen_scan = true;
                    return Ok(false);
                }
                _ => return Ok(true),
            }
        }
    }

    let mut v = Visitor { seen_scan: false };
    p.accept(&mut v).unwrap();
    return v.seen_scan;
}

pub fn plan_cross_join(
    node: &CrossJoin,
    inputs: &[Arc<dyn ExecutionPlan>],
    ctx_state: &ExecutionContextState,
) -> Result<CrossJoinExec> {
    assert_eq!(inputs.len(), 2);
    let left = &inputs[0];
    let right = &inputs[1];

    let schema =
        build_join_schema(&left.schema(), &right.schema(), &[], &JoinType::Left)?;
    let on = DefaultPhysicalPlanner::default()
        .create_physical_expr(&node.on, &schema, ctx_state)?;

    Ok(CrossJoinExec {
        left: left.clone(),
        left_result: Mutex::new(None),
        right: right.clone(),
        on,
        schema: Arc::new(schema),
    })
}

pub struct CrossJoinPlanner;
impl ExtensionPlanner for CrossJoinPlanner {
    fn plan_extension(
        &self,
        node: &dyn UserDefinedLogicalNode,
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx_state: &ExecutionContextState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let node = match node.as_any().downcast_ref::<CrossJoin>() {
            None => return Ok(None),
            Some(j) => j,
        };

        Ok(Some(Arc::new(plan_cross_join(node, inputs, ctx_state)?)))
    }
}

/// This node currently exposes only right as its child.
/// Left is assumed to be constant and **not** exposed as a child of the plan. This is for
/// convenience of the consuming CubeStore code, lifting this restriction should be easy.
#[derive(Debug)]
pub struct CrossJoinExec {
    pub left: Arc<dyn ExecutionPlan>,
    pub left_result: Mutex<Option<std::result::Result<Arc<RecordBatch>, ()>>>,
    pub right: Arc<dyn ExecutionPlan>,
    pub on: Arc<dyn PhysicalExpr>,
    pub schema: DFSchemaRef,
}

impl CrossJoinExec {
    pub fn with_new_children_typed(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> CrossJoinExec {
        assert_eq!(children.len(), 1);
        CrossJoinExec {
            left: self.left.clone(),
            left_result: Mutex::new(None),
            right: children[0].clone(),
            on: self.on.clone(),
            schema: self.schema.clone(),
        }
    }

    async fn do_compute_left(&self) -> Result<RecordBatch> {
        let batches = collect(self.left.clone()).await?;
        let num_rows = batches.iter().map(|b| b.num_rows()).sum();
        Ok(concat_batches(
            &self.left.schema().to_schema_ref(),
            &batches,
            num_rows,
        )?)
    }

    pub async fn compute_left(&self) -> Result<Arc<RecordBatch>> {
        let mut left = self.left_result.lock().await;
        if left.is_none() {
            match self.do_compute_left().await {
                Ok(data) => *left = Some(Ok(Arc::new(data))),
                Err(e) => {
                    *left = Some(Err(()));
                    return Err(e);
                }
            }
        }
        match left.as_ref().unwrap() {
            Ok(data) => Ok(data.clone()),
            Err(()) => Err(DataFusionError::Internal("Could not compute left portion of cross join. See errors from other partitions for details".to_string())),
        }
    }
}

#[async_trait]
impl ExecutionPlan for CrossJoinExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> DFSchemaRef {
        self.schema.clone()
    }

    fn output_partitioning(&self) -> Partitioning {
        self.right.output_partitioning()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.right.clone()]
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(self.with_new_children_typed(children)))
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        let left = self.compute_left().await?;
        let right = self.right.execute(partition).await?;
        let schema = self.schema.to_schema_ref();
        let on = self.on.clone();
        Ok(Box::pin(StreamWithSchema::wrap(
            schema.clone(),
            right.map(move |r| {
                let r = r?;
                let mut batches = Vec::new();
                left_cross_join(&left, &r, &schema, on.as_ref(), |batch, included| {
                    batches.push(filter_record_batch(&batch, included)?);
                    Ok(())
                })
                .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                concat_batches(
                    &schema,
                    &batches,
                    batches.iter().map(|b| b.num_rows()).sum(),
                )
            }),
        )))
    }
}

pub fn left_cross_join(
    left: &RecordBatch,
    right: &RecordBatch,
    join_schema: &SchemaRef,
    join_on: &dyn PhysicalExpr,
    mut on_result: impl FnMut(RecordBatch, /*included*/ &BooleanArray) -> Result<()>,
) -> Result<()> {
    for l in 0..left.num_rows() {
        let indices = UInt64Array::from(vec![l as u64; right.num_rows()]);
        let mut cols = Vec::with_capacity(left.num_columns() + right.num_columns());
        for c in left.columns() {
            cols.push(take(c.as_ref(), &indices, None)?)
        }
        for c in right.columns() {
            cols.push(c.clone())
        }

        let joined = RecordBatch::try_new(join_schema.clone(), cols)?;
        let included = join_on.evaluate(&joined)?.into_array(right.num_rows());
        let included = match included.as_any().downcast_ref::<BooleanArray>() {
            None => {
                return Err(DataFusionError::Execution(
                    "Join predicate returned non-boolean result".to_string(),
                ))
            }
            Some(a) => a,
        };

        on_result(joined, &included)?
    }

    Ok(())
}

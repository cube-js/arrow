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

//! FilterExec evaluates a boolean predicate against all input batches to determine which rows to
//! include in its output batches.

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use super::{RecordBatchStream, SendableRecordBatchStream};
use crate::error::{DataFusionError, Result};
use crate::physical_plan::{ExecutionPlan, OptimizerHints, Partitioning, PhysicalExpr};
use arrow::array::BooleanArray;
use arrow::compute::filter_record_batch;
use arrow::datatypes::{DataType, SchemaRef};
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;

use async_trait::async_trait;

use crate::logical_plan::{DFSchemaRef, Operator};
use crate::physical_plan::expressions::{
    BinaryExpr, CastExpr, Column, Literal, TryCastExpr,
};
use futures::stream::{Stream, StreamExt};

/// FilterExec evaluates a boolean predicate against all input batches to determine which rows to
/// include in its output batches.
#[derive(Debug)]
pub struct FilterExec {
    /// The expression to filter on. This expression must evaluate to a boolean value.
    predicate: Arc<dyn PhysicalExpr>,
    /// The input plan
    input: Arc<dyn ExecutionPlan>,
}

impl FilterExec {
    /// Create a FilterExec on an input
    pub fn try_new(
        predicate: Arc<dyn PhysicalExpr>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Result<Self> {
        match predicate.data_type(input.schema().as_ref())? {
            DataType::Boolean => Ok(Self {
                predicate,
                input: input.clone(),
            }),
            other => Err(DataFusionError::Plan(format!(
                "Filter predicate must return boolean values, not {:?}",
                other
            ))),
        }
    }

    /// The expression to filter on. This expression must evaluate to a boolean value.
    pub fn predicate(&self) -> &Arc<dyn PhysicalExpr> {
        &self.predicate
    }

    /// The input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

#[async_trait]
impl ExecutionPlan for FilterExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Get the schema for this execution plan
    fn schema(&self) -> DFSchemaRef {
        // The filter operator does not make any changes to the schema of its input
        self.input.schema()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        self.input.output_partitioning()
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.len() {
            1 => Ok(Arc::new(FilterExec::try_new(
                self.predicate.clone(),
                children[0].clone(),
            )?)),
            _ => Err(DataFusionError::Internal(
                "FilterExec wrong number of children".to_string(),
            )),
        }
    }

    fn output_hints(&self) -> OptimizerHints {
        let inputs_hints = self.input.output_hints();

        let mut single_value_columns = inputs_hints.single_value_columns;
        let schema = self.schema().to_schema_ref();
        for c in extract_single_value_columns(self.predicate.as_ref()) {
            if let Some(i) = c
                .lookup_field(&schema)
                .ok()
                .and_then(|f| schema.index_of(f.name()).ok())
            {
                single_value_columns.push(i)
            }
        }
        single_value_columns.sort_unstable();
        single_value_columns.dedup();

        OptimizerHints {
            sort_order: inputs_hints.sort_order,
            single_value_columns,
        }
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(FilterExecStream {
            schema: self.input.schema().to_schema_ref(),
            predicate: self.predicate.clone(),
            input: self.input.execute(partition).await?,
        }))
    }
}

fn extract_single_value_columns(predicate: &dyn PhysicalExpr) -> Vec<&Column> {
    let mut columns = Vec::new();
    extract_single_value_columns_impl(predicate, &mut columns);
    columns
}

fn extract_single_value_columns_impl<'a>(
    predicate: &'a dyn PhysicalExpr,
    out: &mut Vec<&'a Column>,
) {
    // TODO: more sophisticated expressions.
    let is_constant = |mut e: &dyn PhysicalExpr| loop {
        if e.as_any().is::<Literal>() {
            return true;
        } else if let Some(c) = e.as_any().downcast_ref::<CastExpr>() {
            e = c.expr().as_ref();
        } else if let Some(c) = e.as_any().downcast_ref::<TryCastExpr>() {
            e = c.expr().as_ref();
        } else {
            return false;
        }
    };

    let predicate = predicate.as_any();
    if let Some(binary) = predicate.downcast_ref::<BinaryExpr>() {
        match binary.op() {
            Operator::And => {
                extract_single_value_columns_impl(binary.left().as_ref(), out);
                extract_single_value_columns_impl(binary.right().as_ref(), out);
            }
            Operator::Eq => {
                let mut left = binary.left();
                let mut right = binary.right();
                if !left.as_any().is::<Column>() {
                    std::mem::swap(&mut left, &mut right);
                }
                let left = left.as_any().downcast_ref::<Column>();
                if left.is_some() && is_constant(right.as_ref()) {
                    out.push(left.unwrap());
                }
            }
            _ => {}
        }
    }
}

/// The FilterExec streams wraps the input iterator and applies the predicate expression to
/// determine which rows to include in its output batches
struct FilterExecStream {
    /// Output schema, which is the same as the input schema for this operator
    schema: SchemaRef,
    /// The expression to filter on. This expression must evaluate to a boolean value.
    predicate: Arc<dyn PhysicalExpr>,
    /// The input partition to filter.
    input: SendableRecordBatchStream,
}

pub(crate) fn batch_filter(
    batch: &RecordBatch,
    predicate: &Arc<dyn PhysicalExpr>,
) -> ArrowResult<RecordBatch> {
    predicate
        .evaluate(&batch)
        .map(|v| v.into_array(batch.num_rows()))
        .map_err(DataFusionError::into_arrow_external_error)
        .and_then(|array| {
            array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal(
                        "Filter predicate evaluated to non-boolean value".to_string(),
                    )
                    .into_arrow_external_error()
                })
                // apply filter array to record batch
                .and_then(|filter_array| filter_record_batch(batch, filter_array))
        })
}

impl Stream for FilterExecStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.input.poll_next_unpin(cx).map(|x| match x {
            Some(Ok(batch)) => Some(batch_filter(&batch, &self.predicate)),
            other => other,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // same number of record batches
        self.input.size_hint()
    }
}

impl RecordBatchStream for FilterExecStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::logical_plan::ToDFSchema;
    use crate::physical_plan::csv::{CsvExec, CsvReadOptions};
    use crate::physical_plan::expressions::*;
    use crate::physical_plan::ExecutionPlan;
    use crate::scalar::ScalarValue;
    use crate::test;
    use crate::{logical_plan::Operator, physical_plan::collect};
    use std::iter::Iterator;

    #[tokio::test]
    async fn simple_predicate() -> Result<()> {
        let schema = test::aggr_test_schema();

        let partitions = 4;
        let path = test::create_partitioned_csv("aggregate_test_100.csv", partitions)?;

        let csv = CsvExec::try_new(
            &path,
            CsvReadOptions::new().schema(&schema),
            None,
            1024,
            None,
        )?;

        let predicate: Arc<dyn PhysicalExpr> = binary(
            binary(
                col("c2"),
                Operator::Gt,
                lit(ScalarValue::from(1u32)),
                &schema.clone().to_dfschema()?,
            )?,
            Operator::And,
            binary(
                col("c2"),
                Operator::Lt,
                lit(ScalarValue::from(4u32)),
                &schema.clone().to_dfschema()?,
            )?,
            &schema.to_dfschema()?,
        )?;

        let filter: Arc<dyn ExecutionPlan> =
            Arc::new(FilterExec::try_new(predicate, Arc::new(csv))?);

        let results = collect(filter).await?;

        results
            .iter()
            .for_each(|batch| assert_eq!(13, batch.num_columns()));
        let row_count: usize = results.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(41, row_count);

        Ok(())
    }
}

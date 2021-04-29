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

//! Defines the OFFSET plan

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::stream::Stream;
use futures::stream::StreamExt;

use crate::error::{DataFusionError, Result};
use crate::physical_plan::{Distribution, ExecutionPlan, OptimizerHints, Partitioning};
use arrow::array::ArrayRef;
use arrow::datatypes::SchemaRef;
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;

use super::{RecordBatchStream, SendableRecordBatchStream};

use crate::logical_plan::DFSchemaRef;
use async_trait::async_trait;

/// Skips first n rows of the input plan
#[derive(Debug)]
pub struct SkipExec {
    /// Input execution plan
    input: Arc<dyn ExecutionPlan>,
    /// Number of rows to skip
    limit: usize,
}

impl SkipExec {
    /// Create a new MergeExec
    pub fn new(input: Arc<dyn ExecutionPlan>, limit: usize) -> Self {
        SkipExec { input, limit }
    }

    /// Input execution plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Maximum number of rows to return
    pub fn limit(&self) -> usize {
        self.limit
    }
}

#[async_trait]
impl ExecutionPlan for SkipExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> DFSchemaRef {
        self.input.schema()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn required_child_distribution(&self) -> Distribution {
        Distribution::SinglePartition
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.len() {
            1 => Ok(Arc::new(SkipExec::new(children[0].clone(), self.limit))),
            _ => Err(DataFusionError::Internal(
                "SkipExec wrong number of children".to_string(),
            )),
        }
    }

    fn output_hints(&self) -> OptimizerHints {
        self.input.output_hints()
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        if 0 != partition {
            return Err(DataFusionError::Internal(format!(
                "SkipExec invalid partition {}",
                partition
            )));
        }

        if 1 != self.input.output_partitioning().partition_count() {
            return Err(DataFusionError::Internal(
                "SkipExec requires a single input partition".to_owned(),
            ));
        }

        let stream = self.input.execute(0).await?;
        Ok(Box::pin(SkipStream::new(stream, self.limit)))
    }
}

/// A Skip stream skips first `skip` rows.
struct SkipStream {
    to_skip: usize,
    input: SendableRecordBatchStream,
    // the current count
    current_skipped: usize,
}

impl SkipStream {
    fn new(input: SendableRecordBatchStream, to_skip: usize) -> Self {
        Self {
            to_skip,
            input,
            current_skipped: 0,
        }
    }

    #[must_use]
    fn consume_batch(&mut self, batch: RecordBatch) -> Option<RecordBatch> {
        let to_skip_rows = self.to_skip - self.current_skipped;
        if to_skip_rows == 0 {
            Some(batch)
        } else if batch.num_rows() <= to_skip_rows {
            self.current_skipped += batch.num_rows();
            None
        } else {
            self.current_skipped = self.to_skip;
            Some(skip_first_rows(&batch, to_skip_rows))
        }
    }
}

fn skip_first_rows(batch: &RecordBatch, n: usize) -> RecordBatch {
    let sliced_columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| c.slice(n, c.len() - n))
        .collect();
    RecordBatch::try_new(batch.schema(), sliced_columns).unwrap()
}

impl Stream for SkipStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        loop {
            let next = match self.input.poll_next_unpin(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(batch))) => batch,
                other => return other,
            };

            match self.consume_batch(next) {
                None => continue, // We are still skipping these rows.
                Some(batch) => return Poll::Ready(Some(Ok(batch))),
            }
        }
    }
}

impl RecordBatchStream for SkipStream {
    /// Get the schema
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }
}

// TODO: tests

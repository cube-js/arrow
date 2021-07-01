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

use crate::cube_ext::stream::StreamWithSchema;
use crate::error::DataFusionError;
use crate::logical_plan::DFSchemaRef;
use crate::physical_plan::{ExecutionPlan, Partitioning, SendableRecordBatchStream};
use async_trait::async_trait;
use futures::{stream, StreamExt, TryStreamExt};
use std::any::Any;
use std::sync::Arc;

/// Reports batches from input strictly in the same order as they arrive.
/// Both partition order and batches order inside partitions is preserved.
#[derive(Debug)]
pub struct SequenceExec {
    pub input: Arc<dyn ExecutionPlan>,
}

#[async_trait]
impl ExecutionPlan for SequenceExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> DFSchemaRef {
        self.input.schema()
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn with_new_children(
        &self,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        assert_eq!(children.len(), 1);
        Ok(Arc::new(SequenceExec {
            input: children.remove(0),
        }))
    }

    async fn execute(
        &self,
        partition: usize,
    ) -> Result<SendableRecordBatchStream, DataFusionError> {
        assert_eq!(partition, 0);
        let n = self.input.output_partitioning().partition_count();
        let input = self.input.clone();
        let s = stream::iter(0..n)
            .then(move |i| {
                let input = input.clone();
                async move {
                    input
                        .execute(i)
                        .await
                        .map_err(|e| e.into_arrow_external_error())
                }
            })
            .try_flatten();
        Ok(Box::pin(StreamWithSchema::wrap(
            self.input.schema().to_schema_ref(),
            s,
        )))
    }
}

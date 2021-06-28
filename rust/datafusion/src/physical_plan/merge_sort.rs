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

//! Merge Sort implementation

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::stream::{Fuse, Stream};
use futures::StreamExt;

pub use arrow::compute::SortOptions;
use arrow::compute::{is_not_null, lexsort_to_indices, take, SortColumn, TakeOptions};
use arrow::datatypes::SchemaRef;
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;
use arrow::{array::ArrayRef, error::ArrowError};

use super::{RecordBatchStream, SendableRecordBatchStream};
use crate::error::{DataFusionError, Result};
use crate::physical_plan::expressions::if_then_else;
use crate::physical_plan::{ExecutionPlan, OptimizerHints, Partitioning};

use crate::logical_plan::DFSchemaRef;
use crate::physical_plan::memory::MemoryStream;
use arrow::compute::kernels::merge::merge_sort_indices;
use async_trait::async_trait;
use futures::future::join_all;

/// Sort execution plan
#[derive(Debug)]
pub struct MergeSortExec {
    input: Arc<dyn ExecutionPlan>,
    /// Columns to sort on
    pub columns: Vec<String>,
}

impl MergeSortExec {
    /// Create a new sort execution plan
    pub fn try_new(input: Arc<dyn ExecutionPlan>, columns: Vec<String>) -> Result<Self> {
        if columns.is_empty() {
            return Err(DataFusionError::Internal(
                "Empty columns passed for MergeSortExec".to_string(),
            ));
        }
        Ok(Self { input, columns })
    }

    /// Input execution plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }
}

#[async_trait]
impl ExecutionPlan for MergeSortExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> DFSchemaRef {
        self.input.schema()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(MergeSortExec::try_new(
            children[0].clone(),
            self.columns.clone(),
        )?))
    }

    fn output_hints(&self) -> OptimizerHints {
        OptimizerHints {
            single_value_columns: self.input.output_hints().single_value_columns,
            sort_order: self
                .columns
                .iter()
                .map(|c| self.schema().index_of(&c).ok())
                .collect(),
        }
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        if 0 != partition {
            return Err(DataFusionError::Internal(format!(
                "MergeSortExec invalid partition {}",
                partition
            )));
        }

        let inputs = join_all(
            (0..self.input.output_partitioning().partition_count())
                .map(|i| self.input.execute(i))
                .collect::<Vec<_>>(),
        )
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

        if inputs.len() == 1 {
            return Ok(inputs.into_iter().next().unwrap());
        }

        Ok(Box::pin(MergeSortStream::new(
            self.input.schema().to_schema_ref(),
            inputs,
            self.columns.clone(),
        )))
    }
}

/// Sort execution plan to resort merge join results
#[derive(Debug)]
pub struct MergeReSortExec {
    input: Arc<dyn ExecutionPlan>,
    columns: Vec<String>,
}

impl MergeReSortExec {
    /// Create a new sort execution plan
    pub fn try_new(input: Arc<dyn ExecutionPlan>, columns: Vec<String>) -> Result<Self> {
        Ok(Self { input, columns })
    }
}

#[async_trait]
impl ExecutionPlan for MergeReSortExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> DFSchemaRef {
        self.input.schema()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(MergeReSortExec::try_new(
            children[0].clone(),
            self.columns.clone(),
        )?))
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        if 0 != partition {
            return Err(DataFusionError::Internal(format!(
                "MergeReSortExec invalid partition {}",
                partition
            )));
        }

        if 1 != self.input.output_partitioning().partition_count() {
            return Err(DataFusionError::Internal(format!(
                "MergeReSortExec expects only one partition but got {}",
                self.input.output_partitioning().partition_count()
            )));
        }

        let stream = self.input.execute(0).await?;
        let all_batches = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<ArrowResult<Vec<_>>>()?;

        let schema = self.input.schema().to_schema_ref();
        let sorted_batches = all_batches
            .into_iter()
            .map(|b| -> Result<SendableRecordBatchStream> {
                Ok(Box::pin(MemoryStream::try_new(
                    vec![sort_batch(&self.columns, &schema, b)?],
                    schema.clone(),
                    None,
                )?))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Box::pin(MergeSortStream::new(
            self.input.schema().to_schema_ref(),
            sorted_batches,
            self.columns.clone(),
        )))
    }
}

fn sort_batch(
    columns: &Vec<String>,
    schema: &SchemaRef,
    batch: RecordBatch,
) -> ArrowResult<RecordBatch> {
    let columns_to_sort = columns
        .iter()
        .map(|c| -> ArrowResult<SortColumn> {
            Ok(SortColumn {
                values: batch.column(batch.schema().index_of(c)?).clone(),
                options: None,
            })
        })
        .collect::<ArrowResult<Vec<_>>>()?;
    let indices = lexsort_to_indices(columns_to_sort.as_slice(), None)?;

    RecordBatch::try_new(
        schema.clone(),
        batch
            .columns()
            .iter()
            .map(|column| {
                take(
                    column.as_ref(),
                    &indices,
                    // disable bound check overhead since indices are already generated from
                    // the same record batch
                    Some(TakeOptions {
                        check_bounds: false,
                    }),
                )
            })
            .collect::<ArrowResult<Vec<ArrayRef>>>()?,
    )
}

struct MergeSortStream {
    schema: SchemaRef,
    columns: Vec<String>,
    poll_states: Vec<MergeSortStreamState>,
}

impl MergeSortStream {
    fn new(
        schema: SchemaRef,
        inputs: Vec<SendableRecordBatchStream>,
        columns: Vec<String>,
    ) -> Self {
        Self {
            schema,
            columns,
            poll_states: inputs
                .into_iter()
                .map(|stream| MergeSortStreamState::new(stream))
                .collect(),
        }
    }
}

struct MergeSortStreamState {
    stream: Fuse<SendableRecordBatchStream>,
    poll_state: Poll<Option<ArrowResult<(usize, RecordBatch)>>>,
}

impl MergeSortStreamState {
    fn new(stream: SendableRecordBatchStream) -> Self {
        Self {
            stream: stream.fuse(),
            poll_state: Poll::Pending,
        }
    }

    pub fn update_state(&mut self, cx: &mut std::task::Context<'_>) {
        if !self.poll_state.is_pending() {
            return;
        }
        let inner = self.stream.poll_next_unpin(cx);
        match inner {
            // skip empty batches and wait for the next poll.
            Poll::Ready(Some(Ok(b))) if b.num_rows() == 0 => {
                cx.waker().wake_by_ref();
                return;
            }
            _ => {}
        }
        self.poll_state = inner.map(|option| match option {
            Some(batch) => Some(batch.map(|b| (0, b))),
            None => None,
        });
    }

    pub fn take_batch(&mut self) -> Option<ArrowResult<(usize, RecordBatch)>> {
        let mut res = Poll::Pending;
        std::mem::swap(&mut res, &mut self.poll_state);
        if let Poll::Ready(option) = &mut res {
            option.take()
        } else {
            panic!(
                "Invalid merge sort state: unexpected empty state: {:?}",
                self.poll_state
            );
        }
    }

    pub fn update_batch(&mut self, new_cursor: usize, batch: RecordBatch) {
        if let Poll::Ready(_) = self.poll_state {
            panic!(
                "Invalid merge sort state: unexpected ready state: {:?}",
                self.poll_state
            );
        } else {
            self.poll_state = if new_cursor == batch.num_rows() {
                Poll::Pending
            } else {
                Poll::Ready(Some(Ok((new_cursor, batch))))
            }
        }
    }
}

impl Stream for MergeSortStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        for state in self.poll_states.iter_mut() {
            state.update_state(cx);
        }

        if self.poll_states.iter().all(|s| s.poll_state.is_ready()) {
            let res = self
                .poll_states
                .iter_mut()
                .map(|s| s.take_batch().transpose())
                .collect::<ArrowResult<Vec<_>>>()
                .and_then(|all_batches| -> ArrowResult<Option<RecordBatch>> {
                    let mut batches = Vec::with_capacity(all_batches.len());
                    let mut batch_indices = Vec::with_capacity(all_batches.len());
                    for (i, b) in all_batches.into_iter().enumerate() {
                        if let Some(b) = b {
                            batch_indices.push(i);
                            batches.push(b);
                        }
                    }
                    if batches.is_empty() {
                        return Ok(None);
                    }
                    let (new_cursors, sorted_batch) = merge_sort(
                        batches.iter().map(|(c, b)| (*c, b)).collect(),
                        self.columns.clone(),
                    )?;

                    assert_eq!(new_cursors.len(), batches.len());
                    for (i, b) in batches.into_iter().enumerate() {
                        self.poll_states[batch_indices[i]]
                            .update_batch(new_cursors[i], b.1);
                    }

                    Ok(Some(sorted_batch))
                });

            Poll::Ready(res.transpose())
        } else {
            Poll::Pending
        }
    }
}

fn merge_sort(
    batches: Vec<(usize, &RecordBatch)>,
    columns: Vec<String>,
) -> ArrowResult<(Vec<usize>, RecordBatch)> {
    let lasts = (0..batches.len()).map(|_| false).collect::<Vec<_>>();
    let cursors = batches.iter().map(|(c, _)| *c).collect::<Vec<_>>();
    let arrays = batches
        .iter()
        .map(|(_, batch)| {
            columns
                .iter()
                .map(|c| -> ArrowResult<ArrayRef> {
                    Ok(batch.column(batch.schema().index_of(c)?).clone())
                })
                .collect::<ArrowResult<Vec<_>>>()
        })
        .collect::<ArrowResult<Vec<_>>>()?;
    let indices = merge_sort_indices(
        arrays
            .iter()
            .map(|cols| cols.as_slice())
            .collect::<Vec<_>>(),
        cursors,
        lasts,
    )?;
    let columns_to_coalesce = batches
        .iter()
        .zip(indices.iter())
        .map(|((_, batch), (_, i))| -> ArrowResult<Vec<ArrayRef>> {
            batch
                .columns()
                .iter()
                .map(|c| take(c.as_ref(), i, None))
                .collect::<ArrowResult<Vec<_>>>()
        })
        .collect::<ArrowResult<Vec<_>>>()?;
    let new_batch = RecordBatch::try_new(
        batches[0].1.schema(),
        (0..batches[0].1.columns().len())
            .map(|column_index| {
                let mut column_arrays = columns_to_coalesce
                    .iter()
                    .map(|batch_columns| batch_columns[column_index].clone());
                let first = Ok(column_arrays.next().unwrap());
                column_arrays.fold(first, |res, b| {
                    res.and_then(|a| -> ArrowResult<ArrayRef> {
                        Ok(if_then_else(
                            &is_not_null(a.as_ref())?,
                            a.clone(),
                            b,
                            a.data_type(),
                        )
                        .map_err(|e| ArrowError::ComputeError(e.to_string()))?)
                    })
                })
            })
            .collect::<ArrowResult<Vec<_>>>()?,
    )?;
    assert_eq!(
        new_batch.num_rows(),
        batches
            .iter()
            .zip(indices.iter())
            .map(|((offset, _), (new_offset, _))| new_offset - offset)
            .sum::<usize>()
    );
    Ok((indices.iter().map(|(i, _)| *i).collect(), new_batch))
}

impl RecordBatchStream for MergeSortStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical_plan::collect;
    use crate::physical_plan::memory::MemoryExec;
    use arrow::array::*;
    use arrow::datatypes::*;
    use itertools::Itertools;

    #[tokio::test]
    async fn two_inputs_three_batches() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::UInt32, true),
            Field::new("b", DataType::UInt64, true),
        ]));

        // define data.
        let batch1_1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![
                    None,
                    None,
                    Some(1),
                    Some(1),
                    Some(3),
                    Some(5),
                    Some(5),
                ])),
                Arc::new(UInt64Array::from(vec![
                    Some(1),
                    Some(2),
                    Some(1),
                    Some(2),
                    Some(2),
                    None,
                    Some(2),
                ])),
            ],
        )?;

        let batch1_2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![
                    Some(7),
                    Some(8),
                    Some(8),
                    Some(8),
                    Some(9),
                ])),
                Arc::new(UInt64Array::from(vec![
                    Some(1),
                    Some(2),
                    Some(2),
                    Some(3),
                    None,
                ])),
            ],
        )?;

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![Some(3), Some(5), Some(10)])),
                Arc::new(UInt64Array::from(vec![Some(2), Some(2), None])),
            ],
        )?;

        let sort_exec = Arc::new(MergeSortExec::try_new(
            Arc::new(MemoryExec::try_new(
                &vec![vec![batch1_1, batch1_2], vec![batch2]],
                schema.clone(),
                None,
            )?),
            vec!["a".to_string(), "b".to_string()],
        )?);

        assert_eq!(DataType::UInt32, *sort_exec.schema().field(0).data_type());
        assert_eq!(DataType::UInt64, *sort_exec.schema().field(1).data_type());

        let result: Vec<RecordBatch> = collect(sort_exec).await?;
        assert_eq!(result.len(), 3);

        assert_eq!(
            vec![
                (None, Some("1".to_owned())),
                (None, Some("2".to_owned())),
                (Some("1".to_owned()), Some("1".to_owned())),
                (Some("1".to_owned()), Some("2".to_owned())),
                (Some("3".to_owned()), Some("2".to_owned())),
                (Some("3".to_owned()), Some("2".to_owned())),
                (Some("5".to_owned()), None),
                (Some("5".to_owned()), Some("2".to_owned())),
                (Some("5".to_owned()), Some("2".to_owned())),
            ],
            transform_batch_for_assert(&result[0])
        );

        assert_eq!(
            vec![
                (Some("7".to_owned()), Some("1".to_owned())),
                (Some("8".to_owned()), Some("2".to_owned())),
                (Some("8".to_owned()), Some("2".to_owned())),
                (Some("8".to_owned()), Some("3".to_owned())),
                (Some("9".to_owned()), None),
            ],
            transform_batch_for_assert(&result[1])
        );

        assert_eq!(
            vec![(Some("10".to_owned()), None),],
            transform_batch_for_assert(&result[2])
        );

        Ok(())
    }

    #[tokio::test]
    async fn resort() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::UInt32, true),
            Field::new("b", DataType::UInt64, true),
        ]));

        // define data.
        let batch1_1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![
                    None,
                    None,
                    Some(1),
                    Some(1),
                    Some(3),
                    Some(5),
                    Some(5),
                ])),
                Arc::new(UInt64Array::from(vec![
                    Some(1),
                    Some(2),
                    Some(1),
                    Some(2),
                    Some(2),
                    None,
                    Some(2),
                ])),
            ],
        )?;

        let batch1_2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![
                    Some(7),
                    Some(8),
                    Some(8),
                    Some(8),
                    Some(9),
                ])),
                Arc::new(UInt64Array::from(vec![
                    Some(1),
                    Some(2),
                    Some(2),
                    Some(3),
                    None,
                ])),
            ],
        )?;

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![Some(3), Some(5), Some(10)])),
                Arc::new(UInt64Array::from(vec![Some(2), Some(2), None])),
            ],
        )?;

        let sort_exec = Arc::new(MergeReSortExec::try_new(
            Arc::new(MemoryExec::try_new(
                &vec![vec![batch1_2, batch1_1, batch2]],
                schema.clone(),
                None,
            )?),
            vec!["a".to_string(), "b".to_string()],
        )?);

        assert_eq!(DataType::UInt32, *sort_exec.schema().field(0).data_type());
        assert_eq!(DataType::UInt64, *sort_exec.schema().field(1).data_type());

        let result: Vec<RecordBatch> = collect(sort_exec).await?;
        assert_eq!(result.len(), 3);

        assert_eq!(
            vec![
                (None, Some("1".to_owned())),
                (None, Some("2".to_owned())),
                (Some("1".to_owned()), Some("1".to_owned())),
                (Some("1".to_owned()), Some("2".to_owned())),
                (Some("3".to_owned()), Some("2".to_owned())),
                (Some("3".to_owned()), Some("2".to_owned())),
                (Some("5".to_owned()), None),
                (Some("5".to_owned()), Some("2".to_owned())),
                (Some("5".to_owned()), Some("2".to_owned())),
            ],
            transform_batch_for_assert(&result[0])
        );

        assert_eq!(
            vec![
                (Some("7".to_owned()), Some("1".to_owned())),
                (Some("8".to_owned()), Some("2".to_owned())),
                (Some("8".to_owned()), Some("2".to_owned())),
                (Some("8".to_owned()), Some("3".to_owned())),
                (Some("9".to_owned()), None),
            ],
            transform_batch_for_assert(&result[1])
        );

        assert_eq!(
            vec![(Some("10".to_owned()), None),],
            transform_batch_for_assert(&result[2])
        );

        Ok(())
    }

    #[tokio::test]
    async fn empty_batches() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::UInt32, true),
            Field::new("b", DataType::UInt64, true),
        ]));

        // define data.
        let batch1_1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(Vec::<u32>::new())),
                Arc::new(UInt64Array::from(Vec::<u64>::new())),
            ],
        )?;

        let batch1_2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![
                    Some(7),
                    Some(8),
                    Some(8),
                    Some(8),
                    Some(9),
                ])),
                Arc::new(UInt64Array::from(vec![
                    Some(1),
                    Some(2),
                    Some(2),
                    Some(3),
                    None,
                ])),
            ],
        )?;

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(Vec::<u32>::new())),
                Arc::new(UInt64Array::from(Vec::<u64>::new())),
            ],
        )?;

        let sort_exec = Arc::new(MergeSortExec::try_new(
            Arc::new(MemoryExec::try_new(
                &vec![
                    vec![batch2.clone()],
                    vec![batch1_1.clone(), batch1_1, batch1_2],
                ],
                schema.clone(),
                None,
            )?),
            vec!["a".to_string(), "b".to_string()],
        )?);

        assert_eq!(DataType::UInt32, *sort_exec.schema().field(0).data_type());
        assert_eq!(DataType::UInt64, *sort_exec.schema().field(1).data_type());

        let result: Vec<RecordBatch> = collect(sort_exec).await?;
        assert_eq!(result.len(), 1);
        assert_eq!(
            transform_batch_for_assert(&result[0]),
            vec![
                (Some("7".to_owned()), Some("1".to_owned())),
                (Some("8".to_owned()), Some("2".to_owned())),
                (Some("8".to_owned()), Some("2".to_owned())),
                (Some("8".to_owned()), Some("3".to_owned())),
                (Some("9".to_owned()), None),
            ]
        );

        Ok(())
    }

    fn ints_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]))
    }

    fn ints(d: Vec<i64>) -> RecordBatch {
        RecordBatch::try_new(ints_schema(), vec![Arc::new(Int64Array::from(d))]).unwrap()
    }

    fn to_ints(rs: Vec<RecordBatch>) -> Vec<Vec<i64>> {
        rs.into_iter()
            .map(|r| {
                r.columns()[0]
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    #[tokio::test]
    async fn multiple_inputs_order() {
        let p1 = vec![ints(vec![1, 3])];
        let p2 = vec![ints(vec![2, 4, 6]), ints(vec![8, 9])];
        let p3 = vec![ints(vec![5, 7, 10])];

        let inp = Arc::new(
            MemoryExec::try_new(&vec![p1, p2, p3], ints_schema(), None).unwrap(),
        );
        let r = collect(Arc::new(
            MergeSortExec::try_new(inp, vec!["a".to_string()]).unwrap(),
        ))
        .await
        .unwrap();
        assert_eq!(
            to_ints(r),
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9], vec![10]]
        );
    }

    #[tokio::test]
    async fn empty_batches_2() {
        let p1 = vec![ints(vec![1, 2])];
        let p2 = vec![ints(vec![]), ints(vec![0])];

        let inp =
            Arc::new(MemoryExec::try_new(&vec![p1, p2], ints_schema(), None).unwrap());
        let r = collect(Arc::new(
            MergeSortExec::try_new(inp, vec!["a".to_string()]).unwrap(),
        ))
        .await
        .unwrap();
        assert_eq!(
            to_ints(r).into_iter().flatten().collect_vec(),
            vec![0, 1, 2],
        );
    }

    fn transform_batch_for_assert(
        batch: &RecordBatch,
    ) -> Vec<(Option<String>, Option<String>)> {
        let columns = batch.columns();

        assert_eq!(DataType::UInt32, *columns[0].data_type());
        assert_eq!(DataType::UInt64, *columns[1].data_type());

        let a = as_primitive_array::<UInt32Type>(&columns[0]);
        let b = as_primitive_array::<UInt64Type>(&columns[1]);

        // convert result to strings to allow comparing to expected result containing NaN
        let result: Vec<(Option<String>, Option<String>)> = (0..batch.num_rows())
            .map(|i| {
                let aval = if a.is_valid(i) {
                    Some(a.value(i).to_string())
                } else {
                    None
                };
                let bval = if b.is_valid(i) {
                    Some(b.value(i).to_string())
                } else {
                    None
                };
                (aval, bval)
            })
            .collect();
        result
    }
}

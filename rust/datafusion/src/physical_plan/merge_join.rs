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

//! Defines the join plan for executing partitions in parallel and then joining the results
//! into a set of partitions.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use futures::{Stream, StreamExt};

use arrow::array::{ArrayRef, UInt32Array};
use arrow::datatypes::{Schema, SchemaRef};
use arrow::error::Result as ArrowResult;
use arrow::record_batch::RecordBatch;

use super::{
    hash_utils::{build_join_schema, check_join_is_valid, JoinOn, JoinType},
    merge::MergeExec,
};
use crate::error::{DataFusionError, Result};

use super::{ExecutionPlan, Partitioning, RecordBatchStream, SendableRecordBatchStream};
use arrow::compute::kernels::merge::{merge_join_indices, MergeJoinType};
use arrow::compute::{concat, take};
use std::task::Poll;

/// join execution plan executes partitions in parallel and combines them into a set of
/// partitions.
#[derive(Debug)]
pub struct MergeJoinExec {
    /// left (build) side which gets hashed
    left: Arc<dyn ExecutionPlan>,
    /// right (probe) side which are filtered by the hash table
    right: Arc<dyn ExecutionPlan>,
    /// Set of common columns used to join on
    on: Vec<(String, String)>,
    /// How the join is performed
    join_type: JoinType,
    /// The schema once the join is applied
    schema: SchemaRef,
}

impl MergeJoinExec {
    /// Tries to create a new [MergeJoinExec].
    /// # Error
    /// This function errors when it is not possible to join the left and right sides on keys `on`.
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: &JoinOn,
        join_type: &JoinType,
    ) -> Result<Self> {
        let left_schema = left.schema();
        let right_schema = right.schema();
        check_join_is_valid(&left_schema, &right_schema, &on)?;

        let schema = Arc::new(build_join_schema(
            &left_schema,
            &right_schema,
            on,
            &join_type,
        ));

        let on = on
            .iter()
            .map(|(l, r)| (l.to_string(), r.to_string()))
            .collect();

        Ok(Self {
            left,
            right,
            on,
            join_type: *join_type,
            schema,
        })
    }
}

#[async_trait]
impl ExecutionPlan for MergeJoinExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.left.clone(), self.right.clone()]
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.len() {
            2 => Ok(Arc::new(MergeJoinExec::try_new(
                children[0].clone(),
                children[1].clone(),
                &self.on,
                &self.join_type,
            )?)),
            _ => Err(DataFusionError::Internal(
                "MergeJoinExec wrong number of children".to_string(),
            )),
        }
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "Invalid partition: {}. Expected to be 0",
                partition
            )));
        }

        let merge_left = MergeExec::new(self.left.clone());
        let stream_left = merge_left.execute(0).await?;

        let merge_right = MergeExec::new(self.right.clone());
        let stream_right = merge_right.execute(0).await?;

        let on_left = self.on.iter().map(|on| on.0.clone()).collect::<Vec<_>>();
        let on_right = self.on.iter().map(|on| on.1.clone()).collect::<Vec<_>>();

        Ok(Box::pin(MergeJoinStream {
            schema: self.schema.clone(),
            on_left,
            on_right,
            join_type: self.join_type,
            left: MergeJoinStreamState::new(stream_left),
            right: MergeJoinStreamState::new(stream_right),
        }))
    }
}

/// A stream that issues [RecordBatch]es as they arrive from the right  of the join.
struct MergeJoinStream {
    /// Input schema
    schema: Arc<Schema>,
    /// type of the join
    join_type: JoinType,
    /// columns from the left
    on_left: Vec<String>,
    /// columns from the right
    on_right: Vec<String>,
    /// left
    left: MergeJoinStreamState,
    /// right
    right: MergeJoinStreamState,
}

impl RecordBatchStream for MergeJoinStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

struct MergeJoinStreamState {
    stream: SendableRecordBatchStream,
    current: Option<(usize, RecordBatch)>,
    to_concat: Option<(usize, RecordBatch)>,
    is_last: bool,
}

impl MergeJoinStreamState {
    fn new(stream: SendableRecordBatchStream) -> Self {
        Self {
            stream,
            current: None,
            to_concat: None,
            is_last: false,
        }
    }

    pub fn update_state(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Option<std::task::Poll<Option<ArrowResult<RecordBatch>>>> {
        if self.current.is_none() && !self.is_last {
            if let Poll::Ready(option) = self.stream.poll_next_unpin(cx) {
                match option {
                    Some(res) => match res {
                        Ok(batch) => {
                            let mut batches = vec![(0usize, batch)];
                            if self.to_concat.is_some() {
                                batches.insert(0, self.to_concat.take().unwrap());
                            }
                            match concat_batches(batches) {
                                Ok(concat) => self.current = Some(concat),
                                Err(e) => return Some(Poll::Ready(Some(Err(e)))),
                            }
                        }
                        Err(e) => {
                            return Some(Poll::Ready(Some(Err(e))));
                        }
                    },
                    None => {
                        self.is_last = true;
                        if self.to_concat.is_some() {
                            self.current = self.to_concat.take();
                        } else {
                            return Some(Poll::Ready(None));
                        }
                    }
                }
            } else {
                return Some(Poll::Pending);
            }
        }
        None
    }
}

impl Stream for MergeJoinStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if let Some(r) = self.left.update_state(cx) {
            return r;
        }
        if let Some(r) = self.right.update_state(cx) {
            return r;
        }

        match (self.left.current.clone(), self.right.current.clone()) {
            (Some((left_cursor, left)), Some((right_cursor, right))) => {
                let merge_result = merge_join(
                    self.schema.clone(),
                    &left,
                    &right,
                    &self.on_left,
                    &self.on_right,
                    self.left.is_last,
                    self.right.is_last,
                    left_cursor,
                    right_cursor,
                    &self.join_type,
                );
                Poll::Ready(Some(merge_result.map(
                    |(new_left_cursor, new_right_cursor, batch)| {
                        if new_left_cursor < left.num_rows()
                            && new_right_cursor < right.num_rows()
                        {
                            self.left.to_concat = Some((new_left_cursor, left));
                            self.right.to_concat = Some((new_right_cursor, right));
                            self.left.current = None;
                            self.right.current = None;
                        } else {
                            if new_left_cursor == left.num_rows() {
                                self.left.current = None;
                            } else {
                                self.left.current = Some((new_left_cursor, left));
                            }

                            if new_right_cursor == right.num_rows() {
                                self.right.current = None;
                            } else {
                                self.right.current = Some((new_right_cursor, right));
                            }
                        }
                        batch
                    },
                )))
            }
            (None, Some(_)) => {
                if self.left.is_last {
                    loop {
                        self.right.current = None;
                        if let Some(r) = self.right.update_state(cx) {
                            return r;
                        }
                        if self.right.current.is_none() && self.right.is_last {
                            return Poll::Ready(None);
                        }
                    }
                }
                Poll::Pending
            }
            (Some(_), None) => {
                if self.right.is_last {
                    loop {
                        self.left.current = None;
                        if let Some(r) = self.left.update_state(cx) {
                            return r;
                        }
                        if self.left.current.is_none() && self.left.is_last {
                            return Poll::Ready(None);
                        }
                    }
                }
                Poll::Pending
            }
            (None, None) => {
                if self.left.is_last && self.right.is_last {
                    Poll::Ready(None)
                } else {
                    Poll::Pending
                }
            }
        }
    }
}

/// Concat multiple batches into single one using cursor as offset for concatenation start
pub fn concat_batches(
    batches: Vec<(usize, RecordBatch)>,
) -> ArrowResult<(usize, RecordBatch)> {
    assert!(!batches.is_empty(), "Empty batch provided");
    if batches.len() == 1 {
        return Ok(batches.into_iter().next().unwrap());
    }
    let columns = (0..batches[0].1.num_columns())
        .map(|column_index| -> ArrowResult<ArrayRef> {
            let columns = batches
                .iter()
                .map(|(cursor, batch)| {
                    take(
                        batch.column(column_index).as_ref(),
                        &UInt32Array::from(
                            (*cursor..batch.num_rows())
                                .map(|i| i as u32)
                                .collect::<Vec<u32>>(),
                        ),
                        None,
                    )
                })
                .collect::<ArrowResult<Vec<_>>>()?;
            let concat_array = concat(
                columns
                    .iter()
                    .map(|a| a.as_ref())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )?;
            Ok(concat_array)
        })
        .collect::<ArrowResult<Vec<_>>>()?;
    Ok((0, RecordBatch::try_new(batches[0].1.schema(), columns)?))
}

#[allow(clippy::too_many_arguments)]
fn merge_join(
    schema: SchemaRef,
    left: &RecordBatch,
    right: &RecordBatch,
    on_left: &Vec<String>,
    on_right: &Vec<String>,
    last_left: bool,
    last_right: bool,
    left_cursor: usize,
    right_cursor: usize,
    join_type: &JoinType,
) -> ArrowResult<(usize, usize, RecordBatch)> {
    let ((new_left_cursor, left_indices), (new_right_cursor, right_indices)) =
        merge_join_indices(
            on_left
                .iter()
                .map(|column_name| -> ArrowResult<ArrayRef> {
                    Ok(left.column(left.schema().index_of(column_name)?).clone())
                })
                .collect::<ArrowResult<Vec<_>>>()?
                .as_slice(),
            on_right
                .iter()
                .map(|column_name| -> ArrowResult<ArrayRef> {
                    Ok(right.column(right.schema().index_of(column_name)?).clone())
                })
                .collect::<ArrowResult<Vec<_>>>()?
                .as_slice(),
            left_cursor,
            right_cursor,
            last_left,
            last_right,
            match join_type {
                JoinType::Inner => MergeJoinType::Inner,
                JoinType::Left => MergeJoinType::Left,
                JoinType::Right => MergeJoinType::Right,
            },
        )?;
    let columns = schema
        .fields()
        .iter()
        .map(|field| {
            left.schema()
                .index_of(field.name())
                .and_then(|i| take(left.column(i).as_ref(), left_indices.as_ref(), None))
                .or_else(|_| {
                    right.schema().index_of(field.name()).and_then(|i| {
                        take(right.column(i).as_ref(), right_indices.as_ref(), None)
                    })
                })
        })
        .collect::<ArrowResult<Vec<_>>>()?;
    let batch = RecordBatch::try_new(schema, columns)?;
    Ok((new_left_cursor, new_right_cursor, batch))
}

#[cfg(test)]
mod tests {

    use crate::{
        physical_plan::{common, memory::MemoryExec},
        test::{build_table_i32, columns, format_batch},
    };

    use super::*;
    use std::sync::Arc;

    fn build_table(
        a: (&str, &Vec<i32>),
        b: (&str, &Vec<i32>),
        c: (&str, &Vec<i32>),
    ) -> Arc<dyn ExecutionPlan> {
        let batch = build_table_i32(a, b, c);
        let schema = batch.schema();
        Arc::new(MemoryExec::try_new(&vec![vec![batch]], schema, None).unwrap())
    }

    fn join(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: &[(&str, &str)],
    ) -> Result<MergeJoinExec> {
        join_with_type(left, right, on, &JoinType::Inner)
    }

    fn join_with_type(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: &[(&str, &str)],
        join_type: &JoinType,
    ) -> Result<MergeJoinExec> {
        let on: Vec<_> = on
            .iter()
            .map(|(l, r)| (l.to_string(), r.to_string()))
            .collect();
        MergeJoinExec::try_new(left, right, &on, join_type)
    }

    fn assert_same_rows(result: &[String], expected: &[&str]) {
        let result = result.iter().map(|s| s.clone()).collect::<Vec<_>>();

        let expected = expected.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(result, expected);
    }

    #[tokio::test]
    async fn join_one() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b1", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = &[("b1", "b1")];

        let join = join(left, right, on)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "c2"]);

        let stream = join.execute(0).await?;
        let batches = common::collect(stream).await?;

        let result = format_batch(&batches[0]);
        let expected = vec!["1,4,7,10,70"];
        assert_same_rows(&result, &expected);

        let result = format_batch(&batches[1]);
        let expected = vec!["2,5,8,20,80", "3,5,9,20,80"];
        assert_same_rows(&result, &expected);

        Ok(())
    }

    #[tokio::test]
    async fn join_one_no_shared_column_names() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a2", &vec![10, 20, 30]),
            ("b2", &vec![4, 5, 6]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = &[("b1", "b2")];

        let join = join(left, right, on)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        let stream = join.execute(0).await?;
        let batches = common::collect(stream).await?;

        let result = format_batch(&batches[0]);
        let expected = vec!["1,4,7,10,4,70"];
        assert_same_rows(&result, &expected);

        let result = format_batch(&batches[1]);
        let expected = vec!["2,5,8,20,5,80", "3,5,9,20,5,80"];
        assert_same_rows(&result, &expected);

        Ok(())
    }

    #[tokio::test]
    async fn join_two() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 2]),
            ("b2", &vec![1, 2, 2]),
            ("c1", &vec![7, 8, 9]),
        );
        let right = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b2", &vec![1, 2, 2]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = &[("a1", "a1"), ("b2", "b2")];

        let join = join(left, right, on)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b2", "c1", "c2"]);

        let stream = join.execute(0).await?;
        let batches = common::collect(stream).await?;
        assert_eq!(batches.len(), 2);

        let result = format_batch(&batches[0]);
        let expected = vec!["1,1,7,70"];
        assert_same_rows(&result, &expected);

        let result = format_batch(&batches[1]);
        let expected = vec!["2,2,8,80", "2,2,9,80"];
        assert_same_rows(&result, &expected);

        Ok(())
    }

    #[tokio::test]
    async fn join_two_left() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 2, 3]),
            ("b2", &vec![1, 2, 2, 3]),
            ("c1", &vec![7, 8, 9, 2]),
        );
        let right = build_table(
            ("a1", &vec![1, 2, 3, 4]),
            ("b2", &vec![1, 2, 2, 4]),
            ("c2", &vec![70, 80, 90, 90]),
        );
        let on = &[("a1", "a1"), ("b2", "b2")];

        let join = join_with_type(left, right, on, &JoinType::Left)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b2", "c1", "c2"]);

        let stream = join.execute(0).await?;
        let batches = common::collect(stream).await?;
        assert_eq!(batches.len(), 1);

        let result = format_batch(&batches[0]);
        let expected = vec!["1,1,7,70", "2,2,8,80", "2,2,9,80", "3,3,2,NULL"];
        assert_same_rows(&result, &expected);

        Ok(())
    }

    /// Test where the left has 2 parts, the right with 1 part => 1 part
    #[tokio::test]
    async fn join_one_two_parts_left() -> Result<()> {
        let batch1 = build_table_i32(
            ("a1", &vec![1, 2]),
            ("b2", &vec![1, 2]),
            ("c1", &vec![7, 8]),
        );
        let batch2 =
            build_table_i32(("a1", &vec![2]), ("b2", &vec![2]), ("c1", &vec![9]));
        let schema = batch1.schema();
        let left = Arc::new(
            MemoryExec::try_new(&vec![vec![batch1], vec![batch2]], schema, None).unwrap(),
        );

        let right = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b2", &vec![1, 2, 2]),
            ("c2", &vec![70, 80, 90]),
        );
        let on = &[("a1", "a1"), ("b2", "b2")];

        let join = join(left, right, on)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b2", "c1", "c2"]);

        let stream = join.execute(0).await?;
        let batches = common::collect(stream).await?;
        assert_eq!(batches.len(), 2);

        let result = format_batch(&batches[0]);
        let expected = vec!["1,1,7,70"];

        assert_same_rows(&result, &expected);

        let result = format_batch(&batches[1]);
        let expected = vec!["2,2,8,80", "2,2,9,80"];

        assert_same_rows(&result, &expected);

        Ok(())
    }

    /// Test where the left has 1 part, the right has 2 parts => 2 parts
    #[tokio::test]
    async fn join_one_two_parts_right() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );

        let batch1 = build_table_i32(
            ("a2", &vec![10, 20]),
            ("b1", &vec![4, 5]),
            ("c2", &vec![70, 80]),
        );
        let batch2 =
            build_table_i32(("a2", &vec![30]), ("b1", &vec![5]), ("c2", &vec![90]));
        let schema = batch1.schema();
        let right = Arc::new(
            MemoryExec::try_new(&vec![vec![batch1], vec![batch2]], schema, None).unwrap(),
        );

        let on = &[("b1", "b1")];

        let join = join(left, right, on)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "c2"]);

        // first part
        let stream = join.execute(0).await?;
        let batches = common::collect(stream).await?;
        assert_eq!(batches.len(), 2);

        let result = format_batch(&batches[0]);
        let expected = vec!["1,4,7,10,70"];
        assert_same_rows(&result, &expected);

        let result = format_batch(&batches[1]);
        let expected = vec!["2,5,8,20,80", "2,5,8,30,90", "3,5,9,20,80", "3,5,9,30,90"];
        assert_same_rows(&result, &expected);

        Ok(())
    }

    #[tokio::test]
    async fn mutltiple_batches() -> Result<()> {
        let left_1 = build_table_i32(
            ("a1", &vec![1, 2, 3]),
            ("b1", &vec![4, 5, 5]), // this has a repetition
            ("c1", &vec![7, 8, 9]),
        );

        let left_2 = build_table_i32(
            ("a1", &vec![2, 3]),
            ("b1", &vec![5, 6]), // this has a repetition
            ("c1", &vec![1, 9]),
        );

        let right_1 = build_table_i32(
            ("a2", &vec![10, 20, 120]),
            ("b1", &vec![4, 5, 5]),
            ("c2", &vec![70, 80, 180]),
        );
        let right_2 = build_table_i32(
            ("a2", &vec![30, 40]),
            ("b1", &vec![5, 6]),
            ("c2", &vec![90, 100]),
        );
        let schema_left = left_1.schema();
        let schema = right_1.schema();

        let left = Arc::new(
            MemoryExec::try_new(&vec![vec![left_1], vec![left_2]], schema_left, None)
                .unwrap(),
        );

        let right = Arc::new(
            MemoryExec::try_new(&vec![vec![right_1], vec![right_2]], schema, None)
                .unwrap(),
        );

        let on = &[("b1", "b1")];

        let join = join(left, right, on)?;

        let columns = columns(&join.schema());
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "c2"]);

        // first part
        let stream = join.execute(0).await?;
        let batches = common::collect(stream).await?;
        assert_eq!(batches.len(), 3);

        let result = format_batch(&batches[0]);
        let expected = vec!["1,4,7,10,70"];
        assert_same_rows(&result, &expected);

        let result = format_batch(&batches[1]);
        let expected = vec![
            "2,5,8,20,80",
            "2,5,8,120,180",
            "2,5,8,30,90",
            "3,5,9,20,80",
            "3,5,9,120,180",
            "3,5,9,30,90",
            "2,5,1,20,80",
            "2,5,1,120,180",
            "2,5,1,30,90",
        ];
        assert_same_rows(&result, &expected);

        let result = format_batch(&batches[2]);
        let expected = vec!["3,6,9,40,100"];
        assert_same_rows(&result, &expected);

        Ok(())
    }
}

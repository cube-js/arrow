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

//! Execution plan for reading Parquet files

use std::any::Any;
use std::fs::File;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::{fmt, thread, result};

use crate::error::{ExecutionError, Result};
use crate::physical_plan::ExecutionPlan;
use crate::physical_plan::{common, Partitioning};
use arrow::datatypes::{Schema, SchemaRef};
use arrow::error::{ArrowError, Result as ArrowResult};
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use parquet::file::reader::{SerializedFileReader, FileReader, RowGroupReader};

use crossbeam::channel::{bounded, Receiver, RecvError, Sender};
use parquet::arrow::{ArrowReader, ParquetFileArrowReader};
use parquet::schema::types::Type;
use parquet::file::metadata::{ParquetMetaData, RowGroupMetaData, FileMetaData};
use parquet::record::reader::RowIter;
use parquet::errors::ParquetError;
use std::fmt::Formatter;

/// Execution plan for scanning a Parquet file
#[derive(Clone)]
pub struct ParquetExec {
    /// Path to directory containing partitioned Parquet files with the same schema
    filenames: Vec<String>,
    /// Schema after projection is applied
    schema: SchemaRef,
    /// Projection for which columns to load
    projection: Vec<usize>,
    /// Batch size
    batch_size: usize,
    row_group_filter: Option<Arc<dyn Fn(&RowGroupMetaData) -> bool + Send + Sync>>
}

impl fmt::Debug for ParquetExec {
    fn fmt(&self, f: &mut Formatter<'_>) -> result::Result<(), fmt::Error> {
        f.write_fmt(format_args!("ParquetExec: {:?} using {:?}, {:?}", self.filenames, self.schema, self.projection))
    }
}

impl ParquetExec {
    /// TODO
    pub fn try_new(path: &str,
                   projection: Option<Vec<usize>>,
                   batch_size: usize) -> Result<Self> {
        Self::try_new_with_filter(path, projection, batch_size, None)
    }

    /// Create a new Parquet reader execution plan
    pub fn try_new_with_filter(
        path: &str,
        projection: Option<Vec<usize>>,
        batch_size: usize,
        row_group_filter: Option<Arc<dyn Fn(&RowGroupMetaData) -> bool + Send + Sync>>
    ) -> Result<Self> {
        let mut filenames: Vec<String> = vec![];
        common::build_file_list(path, &mut filenames, ".parquet")?;
        if filenames.is_empty() {
            Err(ExecutionError::General("No files found".to_string()))
        } else {
            let file = File::open(&filenames[0])?;
            let file_reader = Rc::new(SerializedFileReader::new(file)?);
            let mut arrow_reader = ParquetFileArrowReader::new(file_reader);
            let schema = arrow_reader.get_schema()?;

            let projection = match projection {
                Some(p) => p,
                None => (0..schema.fields().len()).collect(),
            };

            let projected_schema = Schema::new(
                projection
                    .iter()
                    .map(|i| schema.field(*i).clone())
                    .collect(),
            );

            Ok(Self {
                filenames,
                schema: Arc::new(projected_schema),
                projection,
                batch_size,
                row_group_filter
            })
        }
    }
}

impl ExecutionPlan for ParquetExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        // this is a leaf node and has no children
        vec![]
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(self.filenames.len())
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(Arc::new(self.clone()))
        } else {
            Err(ExecutionError::General(format!(
                "Children cannot be replaced in {:?}",
                self
            )))
        }
    }

    fn execute(
        &self,
        partition: usize,
    ) -> Result<Arc<Mutex<dyn RecordBatchReader + Send + Sync>>> {
        // because the parquet implementation is not thread-safe, it is necessary to execute
        // on a thread and communicate with channels
        let (response_tx, response_rx): (
            Sender<ArrowResult<Option<RecordBatch>>>,
            Receiver<ArrowResult<Option<RecordBatch>>>,
        ) = bounded(2);

        let filename = self.filenames[partition].clone();
        let projection = self.projection.clone();
        let batch_size = self.batch_size;
        let row_group_filter = self.row_group_filter.clone();

        thread::spawn(move || {
            if let Err(e) = read_file(&filename, projection, batch_size, response_tx, row_group_filter) {
                println!("Parquet reader thread terminated due to error: {:?}", e);
            }
        });

        let iterator = Arc::new(Mutex::new(ParquetIterator {
            schema: self.schema.clone(),
            response_rx,
        }));

        Ok(iterator)
    }
}

fn send_result(
    response_tx: &Sender<ArrowResult<Option<RecordBatch>>>,
    result: ArrowResult<Option<RecordBatch>>,
) -> Result<()> {
    response_tx
        .send(result)
        .map_err(|e| ExecutionError::ExecutionError(e.to_string()))?;
    Ok(())
}

struct FilteredFileReader {
    file_reader: Rc<dyn FileReader>,
    filtered_row_groups: Vec<usize>,
    filtered_metadata: ParquetMetaData
}

impl FilteredFileReader {
    pub fn new(
        file_reader: Rc<dyn FileReader>,
        row_group_filter: Arc<dyn Fn(&RowGroupMetaData) -> bool + Send + Sync>
    ) -> FilteredFileReader {
        let filtered_row_groups = (0..file_reader.num_row_groups())
            .filter(
                |i| match file_reader.get_row_group(*i) {
                    Ok(group) => row_group_filter(group.metadata()),
                    _ => true
                }
            )
            .collect::<Vec<_>>();
        let file_meta = file_reader.metadata().file_metadata();
        FilteredFileReader {
            file_reader: file_reader.clone(),
            filtered_metadata: ParquetMetaData::new(
                FileMetaData::new(
                    file_meta.version(),
                    file_meta.num_rows(),
                    file_meta.created_by().clone(),
                    file_meta.key_value_metadata().clone(),
                    Rc::new(file_meta.schema().clone()),
                    file_meta.schema_descr_ptr(),
                    file_meta.column_orders().map(|v| v.clone())
                ),
                filtered_row_groups.iter().map(|i| {
                    let group = file_reader.metadata().row_group(*i);
                    RowGroupMetaData::from_thrift(group.schema_descr_ptr(), group.to_thrift()).unwrap()
                }).collect::<Vec<_>>()
            ),
            filtered_row_groups,
        }
    }
}

impl FileReader for FilteredFileReader {
    fn metadata(&self) -> &ParquetMetaData {
        &self.filtered_metadata
    }

    fn num_row_groups(&self) -> usize {
        self.filtered_row_groups.len()
    }

    fn get_row_group(&self, i: usize) -> result::Result<Box<dyn RowGroupReader + '_>, ParquetError> {
        self.file_reader.get_row_group(self.filtered_row_groups[i])
    }

    fn get_row_iter(&self, _: Option<Type>) -> result::Result<RowIter, ParquetError> {
        unimplemented!()
    }
}

fn read_file(
    filename: &str,
    projection: Vec<usize>,
    batch_size: usize,
    response_tx: Sender<ArrowResult<Option<RecordBatch>>>,
    row_group_filter: Option<Arc<dyn Fn(&RowGroupMetaData) -> bool + Send + Sync>>
) -> Result<()> {
    let file = File::open(&filename)?;
    let mut file_reader: Rc<dyn FileReader> = Rc::new(SerializedFileReader::new(file)?);
    if let Some(filter) = row_group_filter {
        file_reader = Rc::new(FilteredFileReader::new(file_reader, filter));
    }
    let mut arrow_reader = ParquetFileArrowReader::new(file_reader);
    let mut batch_reader =
        arrow_reader.get_record_reader_by_columns(projection.clone(), batch_size)?;
    loop {
        match batch_reader.next_batch() {
            Ok(Some(batch)) => send_result(&response_tx, Ok(Some(batch)))?,
            Ok(None) => {
                // finished reading file
                send_result(&response_tx, Ok(None))?;
                break;
            }
            Err(e) => {
                let err_msg =
                    format!("Error reading batch from {}: {}", filename, e.to_string());
                // send error to operator
                send_result(
                    &response_tx,
                    Err(ArrowError::ParquetError(err_msg.clone())),
                )?;
                // terminate thread with error
                return Err(ExecutionError::ExecutionError(err_msg));
            }
        }
    }
    Ok(())
}

struct ParquetIterator {
    schema: SchemaRef,
    response_rx: Receiver<ArrowResult<Option<RecordBatch>>>,
}

impl RecordBatchReader for ParquetIterator {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn next_batch(&mut self) -> ArrowResult<Option<RecordBatch>> {
        match self.response_rx.recv() {
            Ok(batch) => batch,
            // RecvError means receiver has exited and closed the channel
            Err(RecvError) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test() -> Result<()> {
        let testdata =
            env::var("PARQUET_TEST_DATA").expect("PARQUET_TEST_DATA not defined");
        let filename = format!("{}/alltypes_plain.parquet", testdata);
        let parquet_exec = ParquetExec::try_new(&filename, Some(vec![0, 1, 2]), 1024)?;
        assert_eq!(parquet_exec.output_partitioning().partition_count(), 1);

        let results = parquet_exec.execute(0)?;
        let mut results = results.lock().unwrap();
        let batch = results.next_batch()?.unwrap();

        assert_eq!(8, batch.num_rows());
        assert_eq!(3, batch.num_columns());

        let schema = batch.schema();
        let field_names: Vec<&str> =
            schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(vec!["id", "bool_col", "tinyint_col"], field_names);

        let batch = results.next_batch()?;
        assert!(batch.is_none());

        let batch = results.next_batch()?;
        assert!(batch.is_none());

        let batch = results.next_batch()?;
        assert!(batch.is_none());

        Ok(())
    }
}

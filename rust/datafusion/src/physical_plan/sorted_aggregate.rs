use crate::error::{DataFusionError, Result};
use crate::physical_plan::group_scalar::GroupByScalar;
use crate::physical_plan::hash_aggregate::{
    create_accumulators, create_group_by_values, write_group_result_row, AccumulatorSet,
    AggregateMode,
};
use crate::physical_plan::AggregateExpr;
use crate::scalar::ScalarValue;
use arrow::array::{ArrayBuilder, ArrayRef};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use itertools::Itertools;
use smallvec::smallvec;
use smallvec::SmallVec;
use std::sync::Arc;

pub(crate) struct Agg {
    key: SmallVec<[GroupByScalar; 2]>,
    accumulators: AccumulatorSet,
}

pub(crate) struct SortedAggState {
    processed_keys: Vec<Box<dyn ArrayBuilder>>,
    processed_values: Vec<Box<dyn ArrayBuilder>>,
    current_agg: Option<Agg>,
}

// TODO: find a safe alternative.
// This assumes builders are well-behaved.
unsafe impl Send for SortedAggState {}

impl SortedAggState {
    pub fn new() -> SortedAggState {
        SortedAggState {
            processed_keys: Vec::new(),
            processed_values: Vec::new(),
            current_agg: None,
        }
    }

    pub fn finish(
        mut self,
        mode: AggregateMode,
        schema: SchemaRef,
    ) -> arrow::error::Result<RecordBatch> {
        if let Some(agg) = self.current_agg {
            write_group_result_row(
                mode,
                &agg.key,
                &agg.accumulators,
                &mut self.processed_keys,
                &mut self.processed_values,
            )
            .map_err(DataFusionError::into_arrow_external_error)?;
        }
        let columns = self
            .processed_keys
            .into_iter()
            .chain(self.processed_values.into_iter())
            .map(|mut c| c.finish())
            .collect_vec();
        if columns.is_empty() {
            Ok(RecordBatch::new_empty(schema))
        } else {
            RecordBatch::try_new(schema, columns)
        }
    }

    pub fn add_batch(
        &mut self,
        mode: AggregateMode,
        agg_exprs: &Vec<Arc<dyn AggregateExpr>>,
        key_columns: &[ArrayRef],
        aggr_input_values: &[Vec<ArrayRef>],
    ) -> Result<()> {
        assert_ne!(key_columns.len(), 0);
        assert_eq!(aggr_input_values.len(), agg_exprs.len());
        let mut values_buffer = Vec::with_capacity(aggr_input_values.len());

        let num_rows = key_columns[0].len();
        if num_rows == 0 {
            return Ok(());
        }
        if self.current_agg.is_none() {
            let mut key = smallvec![GroupByScalar::Int64(0); key_columns.len()];
            create_group_by_values(key_columns, 0, &mut key)?;
            self.current_agg = Some(Agg {
                key,
                accumulators: create_accumulators(agg_exprs)?,
            });
        }

        let mut row_i = 0;
        while row_i < num_rows {
            let current_agg = self.current_agg.as_mut().unwrap();

            let start = row_i;
            let mut end = start;
            while end < key_columns[0].len()
                && agg_key_equals(&current_agg.key, key_columns, end)?
            {
                end += 1
            }

            if start == end {
                // Start a new group, next iteration will do the actual aggregation.
                write_group_result_row(
                    mode,
                    &current_agg.key,
                    &current_agg.accumulators,
                    &mut self.processed_keys,
                    &mut self.processed_values,
                )?;
                create_group_by_values(key_columns, start, &mut current_agg.key)?;
                current_agg.accumulators = create_accumulators(agg_exprs)?;
            } else {
                // Update the current group.
                for i in 0..aggr_input_values.len() {
                    values_buffer.clear();
                    for inp in &aggr_input_values[i] {
                        values_buffer.push(inp.slice(start, end - start));
                    }
                    match mode {
                        AggregateMode::Partial => {
                            current_agg.accumulators[i].update_batch(&values_buffer)?
                        }
                        AggregateMode::Final => {
                            current_agg.accumulators[i].merge_batch(&values_buffer)?
                        }
                    }
                }
            }

            row_i = end;
        }
        Ok(())
    }
}

fn agg_key_equals(
    key: &[GroupByScalar],
    key_columns: &[ArrayRef],
    row: usize,
) -> Result<bool> {
    assert_eq!(key.len(), key_columns.len());
    for i in 0..key.len() {
        // TODO: do not allocate for strings.
        if ScalarValue::from(&key[i])
            != ScalarValue::try_from_array(&key_columns[i], row)?
        {
            return Ok(false);
        }
    }
    return Ok(true);
}

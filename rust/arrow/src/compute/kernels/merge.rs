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

//! Defines merge sort and join kernels

use std::sync::Arc;

use crate::array::*;
use crate::datatypes::*;
use crate::error::{ArrowError, Result};

use core::fmt;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fmt::{Debug, Display, Formatter};

type JoinCursorAndIndices = (usize, bool, Arc<UInt32Array>);
type CursorAndIndices = (usize, Arc<UInt32Array>);

#[derive(Debug)]
pub enum MergeJoinType {
    Inner,
    Left,
    Right,
}

/// Merge join two arrays
pub fn merge_join_indices<'a>(
    left: &'a [ArrayRef],
    right: &'a [ArrayRef],
    left_cursor: usize,
    right_cursor: usize,
    last_left: bool,
    last_right: bool,
    join_type: MergeJoinType,
) -> Result<(JoinCursorAndIndices, JoinCursorAndIndices)> {
    let arrays: Vec<&'a [ArrayRef]> = vec![left, right];

    let comparators = comparators_for(arrays)?;

    let mut left_indices = Vec::<Option<u32>>::new();
    let mut right_indices = Vec::<Option<u32>>::new();

    let left_size = left[0].len();
    let right_size = right[0].len();

    let mut left_merge_cursor = MergeRowCursor::new(&comparators, 0, left_cursor);
    let mut right_merge_cursor = MergeRowCursor::new(&comparators, 1, right_cursor);

    let mut advance_left = false;
    let mut advance_right = false;

    while left_merge_cursor.within_range() && right_merge_cursor.within_range() {
        left_merge_cursor.check_consistency()?;
        right_merge_cursor.check_consistency()?;
        let ordering = left_merge_cursor.cmp(&right_merge_cursor);
        match ordering {
            Ordering::Equal => {
                let mut left_equal_end = left_merge_cursor.next();
                let mut right_equal_end = right_merge_cursor.next();
                while left_equal_end.row_index < left_size
                    && left_equal_end == right_merge_cursor
                {
                    left_equal_end = left_equal_end.next();
                }
                while right_equal_end.row_index < right_size
                    && left_merge_cursor == right_equal_end
                {
                    right_equal_end = right_equal_end.next();
                }
                if left_merge_cursor.is_valid()
                    && right_merge_cursor.is_valid()
                    && !last_left
                    && !last_right
                    && (left_equal_end.row_index == left_size
                        || right_equal_end.row_index == right_size)
                {
                    advance_left = left_equal_end.row_index == left_size;
                    advance_right = right_equal_end.row_index == right_size;
                    break;
                }
                if left_merge_cursor.is_valid() && right_merge_cursor.is_valid() {
                    for li in left_merge_cursor.row_index..left_equal_end.row_index {
                        for ri in right_merge_cursor.row_index..right_equal_end.row_index
                        {
                            left_indices.push(Some(li as u32));
                            right_indices.push(Some(ri as u32));
                        }
                    }
                } else if let MergeJoinType::Left = join_type {
                    for li in left_merge_cursor.row_index..left_equal_end.row_index {
                        left_indices.push(Some(li as u32));
                        right_indices.push(None);
                    }
                } else if let MergeJoinType::Right = join_type {
                    for ri in right_merge_cursor.row_index..right_equal_end.row_index {
                        left_indices.push(None);
                        right_indices.push(Some(ri as u32));
                    }
                }
                left_merge_cursor = left_equal_end;
                right_merge_cursor = right_equal_end;
            }
            Ordering::Less => {
                if let MergeJoinType::Left = join_type {
                    left_indices.push(Some(left_merge_cursor.row_index as u32));
                    right_indices.push(None);
                }
                left_merge_cursor = left_merge_cursor.next();
            }
            Ordering::Greater => {
                if let MergeJoinType::Right = join_type {
                    left_indices.push(None);
                    right_indices.push(Some(right_merge_cursor.row_index as u32));
                }
                right_merge_cursor = right_merge_cursor.next();
            }
        }
    }

    if last_right {
        while left_merge_cursor.within_range() {
            if let MergeJoinType::Left = join_type {
                left_indices.push(Some(left_merge_cursor.row_index as u32));
                right_indices.push(None);
            }
            left_merge_cursor = left_merge_cursor.next();
        }
    }

    if last_left {
        while right_merge_cursor.within_range() {
            if let MergeJoinType::Right = join_type {
                left_indices.push(None);
                right_indices.push(Some(right_merge_cursor.row_index as u32));
            }
            right_merge_cursor = right_merge_cursor.next();
        }
    }

    let left_indices_array = UInt32Array::from(left_indices);
    let right_indices_array = UInt32Array::from(right_indices);

    Ok((
        (
            left_merge_cursor.row_index,
            advance_left,
            Arc::new(left_indices_array),
        ),
        (
            right_merge_cursor.row_index,
            advance_right,
            Arc::new(right_indices_array),
        ),
    ))
}

pub fn merge_sort_indices(
    arrays: Vec<&[ArrayRef]>,
    cursors: Vec<usize>,
    last_array: Vec<bool>,
) -> Result<Vec<CursorAndIndices>> {
    let arrays_count = cursors.len();

    if arrays_count != cursors.len() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Arrays doesn't match cursors len: {} != {}",
            arrays_count,
            cursors.len()
        )));
    }

    let comparators = comparators_for(arrays)?;

    let mut indices = Vec::new();

    for _ in 0..cursors.len() {
        indices.push(Vec::<Option<u32>>::new());
    }

    let mut merge_cursors = cursors
        .into_iter()
        .enumerate()
        .map(|(array_index, row_index)| {
            Reverse(MergeRowCursor::new(&comparators, array_index, row_index))
        })
        .collect::<BinaryHeap<_>>();

    let mut finished_cursors = Vec::new();

    while merge_cursors.iter().all(|Reverse(c)| c.within_range())
        && !merge_cursors.is_empty()
    {
        if let Some(Reverse(current)) = merge_cursors.pop() {
            current.check_consistency()?;
            for (array_index, indices_array) in indices.iter_mut().enumerate() {
                indices_array.push(if current.array_index == array_index {
                    Some(current.row_index as u32)
                } else {
                    None
                })
            }
            let next = current.next();
            let within_range = next.within_range();
            if within_range {
                merge_cursors.push(Reverse(next));
            } else {
                let array_index = next.array_index;
                finished_cursors.push(next);
                if !last_array[array_index] {
                    break;
                }
            }
        }
    }

    let mut merge_cursors_vec = merge_cursors
        .into_iter()
        .map(|Reverse(c)| c)
        .collect::<Vec<_>>();

    merge_cursors_vec.append(&mut finished_cursors);

    merge_cursors_vec.sort_by_key(|c| c.array_index);

    Ok(merge_cursors_vec
        .into_iter()
        .zip(indices.into_iter())
        .map(|(c, indices_array)| {
            (c.row_index, Arc::new(UInt32Array::from(indices_array)))
        })
        .collect())
}

fn comparators_for<'a>(
    arrays: Vec<&'a [ArrayRef]>,
) -> Result<Vec<Box<dyn ArrayComparator + 'a>>> {
    let mut comparators: Vec<Box<dyn ArrayComparator + 'a>> = Vec::new();
    for column_index in 0..arrays[0].len() {
        let to_compare = arrays.iter().map(|a| &a[column_index]).collect::<Vec<_>>();
        if !to_compare
            .iter()
            .all(|a| a.data_type() == to_compare[0].data_type())
        {
            return Err(ArrowError::SchemaError(format!(
                "Mismatched data types for merge join: {:?}",
                to_compare.iter().map(|c| c.data_type()).collect::<Vec<_>>()
            )));
        }

        let comparator: Box<dyn ArrayComparator + 'a> = match to_compare[0].data_type() {
            DataType::Int8 => Box::new(PrimitiveComparator::<Int8Type>::new(to_compare)),
            DataType::Int16 => {
                Box::new(PrimitiveComparator::<Int16Type>::new(to_compare))
            }
            DataType::Int32 => {
                Box::new(PrimitiveComparator::<Int32Type>::new(to_compare))
            }
            DataType::Int64 => {
                Box::new(PrimitiveComparator::<Int64Type>::new(to_compare))
            }
            DataType::Int64Decimal(0) => {
                Box::new(PrimitiveComparator::<Int64Decimal0Type>::new(to_compare))
            }
            DataType::Int64Decimal(1) => {
                Box::new(PrimitiveComparator::<Int64Decimal1Type>::new(to_compare))
            }
            DataType::Int64Decimal(2) => {
                Box::new(PrimitiveComparator::<Int64Decimal2Type>::new(to_compare))
            }
            DataType::Int64Decimal(3) => {
                Box::new(PrimitiveComparator::<Int64Decimal3Type>::new(to_compare))
            }
            DataType::Int64Decimal(4) => {
                Box::new(PrimitiveComparator::<Int64Decimal4Type>::new(to_compare))
            }
            DataType::Int64Decimal(5) => {
                Box::new(PrimitiveComparator::<Int64Decimal5Type>::new(to_compare))
            }
            DataType::Int64Decimal(10) => {
                Box::new(PrimitiveComparator::<Int64Decimal10Type>::new(to_compare))
            }
            DataType::UInt8 => {
                Box::new(PrimitiveComparator::<UInt8Type>::new(to_compare))
            }
            DataType::UInt16 => {
                Box::new(PrimitiveComparator::<UInt16Type>::new(to_compare))
            }
            DataType::UInt32 => {
                Box::new(PrimitiveComparator::<UInt32Type>::new(to_compare))
            }
            DataType::UInt64 => {
                Box::new(PrimitiveComparator::<UInt64Type>::new(to_compare))
            }
            DataType::Time32(TimeUnit::Second) => {
                Box::new(PrimitiveComparator::<Time32SecondType>::new(to_compare))
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                Box::new(PrimitiveComparator::<Time32MillisecondType>::new(
                    to_compare,
                ))
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                Box::new(PrimitiveComparator::<Time64MicrosecondType>::new(
                    to_compare,
                ))
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                Box::new(PrimitiveComparator::<Time64NanosecondType>::new(to_compare))
            }
            DataType::Timestamp(TimeUnit::Second, _) => {
                Box::new(PrimitiveComparator::<TimestampSecondType>::new(to_compare))
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => Box::new(
                PrimitiveComparator::<TimestampMillisecondType>::new(to_compare),
            ),
            DataType::Timestamp(TimeUnit::Microsecond, _) => Box::new(
                PrimitiveComparator::<TimestampMicrosecondType>::new(to_compare),
            ),
            DataType::Timestamp(TimeUnit::Nanosecond, _) => Box::new(
                PrimitiveComparator::<TimestampNanosecondType>::new(to_compare),
            ),
            DataType::Boolean => Box::new(BooleanComparator::new(to_compare)),
            DataType::Utf8 => Box::new(StringComparator::new(to_compare)),
            t => {
                unimplemented!("Merge operations are not supported for data type {:?}", t)
            }
        };
        comparators.push(comparator);
    }
    Ok(comparators)
}

pub struct MergeRowCursor<'a> {
    comparators: &'a [Box<dyn ArrayComparator + 'a>],
    array_index: usize,
    row_index: usize,
}

impl<'a> MergeRowCursor<'a> {
    fn new(
        comparators: &'a [Box<dyn ArrayComparator + 'a>],
        array_index: usize,
        row_index: usize,
    ) -> Self {
        Self {
            comparators,
            array_index,
            row_index,
        }
    }

    fn next(&self) -> Self {
        Self {
            comparators: self.comparators,
            row_index: self.row_index + 1,
            array_index: self.array_index,
        }
    }

    fn advance(&self, row_index: usize) -> Self {
        Self {
            comparators: self.comparators,
            row_index,
            array_index: self.array_index,
        }
    }

    fn check_consistency(&self) -> Result<()> {
        if self.row_index > 0 {
            let prev_point =
                Self::new(self.comparators, self.array_index, self.row_index - 1);
            if &prev_point > self {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Merge operation on unsorted data: {:?}, {:?}",
                    prev_point, self
                )));
            }
        }
        Ok(())
    }

    fn compare(&self, other: &Self) -> Ordering {
        self.comparators.iter().fold(Ordering::Equal, |a, b| {
            a.then(b.compare(
                self.array_index,
                self.row_index,
                other.array_index,
                other.row_index,
            ))
        })
    }

    fn within_range(&self) -> bool {
        self.comparators
            .iter()
            .all(|c| c.within_range(self.array_index, self.row_index))
    }

    fn is_valid(&self) -> bool {
        self.comparators
            .iter()
            .all(|c| c.is_valid(self.array_index, self.row_index))
    }
}

impl<'a> PartialOrd for MergeRowCursor<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.compare(other))
    }
}

impl<'a> PartialEq for MergeRowCursor<'a> {
    fn eq(&self, other: &MergeRowCursor) -> bool {
        self.compare(other) == Ordering::Equal
    }
}

impl<'a> Eq for MergeRowCursor<'a> {}
impl<'a> Ord for MergeRowCursor<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<'a> Debug for MergeRowCursor<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cursor [array: {}, row: {}, value: {:?}]",
            self.array_index,
            self.row_index,
            self.comparators
                .iter()
                .map(|c| c.debug_value(self.array_index, self.row_index))
                .collect::<Vec<_>>()
        )
    }
}

pub trait ArrayComparator {
    fn compare(
        &self,
        left_array_index: usize,
        left_row_index: usize,
        right_array_index: usize,
        right_row_index: usize,
    ) -> Ordering;

    fn within_range(&self, array_index: usize, row_index: usize) -> bool;

    fn debug_value(&self, array_index: usize, row_index: usize) -> Box<dyn Debug>;

    fn is_valid(&self, array_index: usize, row_index: usize) -> bool;
}

#[derive(Debug)]
pub struct StringComparator<'a> {
    arrays: Vec<&'a StringArray>,
}

impl<'a> StringComparator<'a> {
    fn new(arrays: Vec<&'a ArrayRef>) -> Self {
        Self {
            arrays: arrays
                .into_iter()
                .map(|array| array.as_any().downcast_ref::<StringArray>().unwrap())
                .collect(),
        }
    }
}

impl<'a> ArrayComparator for StringComparator<'a> {
    fn compare(
        &self,
        left_array_index: usize,
        left_row_index: usize,
        right_array_index: usize,
        right_row_index: usize,
    ) -> Ordering {
        let left = self.arrays[left_array_index];
        let right = self.arrays[right_array_index];
        let left_null = left.is_null(left_row_index);
        let right_null = right.is_null(right_row_index);
        if left_null && right_null {
            Ordering::Equal
        } else if right_null {
            Ordering::Greater
        } else if left_null {
            Ordering::Less
        } else {
            let left_value = left.value(left_row_index);
            let right_value = right.value(right_row_index);
            left_value.cmp(&right_value)
        }
    }

    fn within_range(&self, array_index: usize, row_index: usize) -> bool {
        row_index < self.arrays[array_index].len()
    }

    fn debug_value(&self, array_index: usize, row_index: usize) -> Box<dyn Debug> {
        Box::new(self.arrays[array_index].value(row_index).to_string())
    }

    fn is_valid(&self, array_index: usize, row_index: usize) -> bool {
        self.arrays[array_index].is_valid(row_index)
    }
}

#[derive(Debug)]
pub struct BooleanComparator<'a> {
    arrays: Vec<&'a BooleanArray>,
}

impl<'a> BooleanComparator<'a> {
    fn new(arrays: Vec<&'a ArrayRef>) -> Self {
        Self {
            arrays: arrays
                .into_iter()
                .map(|array| array.as_any().downcast_ref::<BooleanArray>().unwrap())
                .collect(),
        }
    }
}

impl<'a> ArrayComparator for BooleanComparator<'a> {
    fn compare(
        &self,
        left_array_index: usize,
        left_row_index: usize,
        right_array_index: usize,
        right_row_index: usize,
    ) -> Ordering {
        let left = self.arrays[left_array_index];
        let right = self.arrays[right_array_index];
        let left_null = left.is_null(left_row_index);
        let right_null = right.is_null(right_row_index);
        if left_null && right_null {
            Ordering::Equal
        } else if right_null {
            Ordering::Greater
        } else if left_null {
            Ordering::Less
        } else {
            let left_value = left.value(left_row_index);
            let right_value = right.value(right_row_index);
            left_value.cmp(&right_value)
        }
    }

    fn within_range(&self, array_index: usize, row_index: usize) -> bool {
        row_index < self.arrays[array_index].len()
    }

    fn debug_value(&self, array_index: usize, row_index: usize) -> Box<dyn Debug> {
        Box::new(self.arrays[array_index].value(row_index))
    }

    fn is_valid(&self, array_index: usize, row_index: usize) -> bool {
        self.arrays[array_index].is_valid(row_index)
    }
}

#[derive(Debug)]
pub struct PrimitiveComparator<'a, T: ArrowPrimitiveType>
where
    T::Native: Ord + Display,
{
    arrays: Vec<&'a PrimitiveArray<T>>,
}

impl<'a, T: ArrowPrimitiveType> PrimitiveComparator<'a, T>
where
    T::Native: Ord + Display,
{
    fn new(arrays: Vec<&'a ArrayRef>) -> Self {
        Self {
            arrays: arrays
                .into_iter()
                .map(|array| array.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap())
                .collect(),
        }
    }
}

impl<'a, T: ArrowPrimitiveType> ArrayComparator for PrimitiveComparator<'a, T>
where
    T::Native: Ord + Display,
{
    fn compare(
        &self,
        left_array_index: usize,
        left_row_index: usize,
        right_array_index: usize,
        right_row_index: usize,
    ) -> Ordering {
        let left = self.arrays[left_array_index];
        let right = self.arrays[right_array_index];
        let left_null = left.is_null(left_row_index);
        let right_null = right.is_null(right_row_index);
        if left_null && right_null {
            Ordering::Equal
        } else if right_null {
            Ordering::Greater
        } else if left_null {
            Ordering::Less
        } else {
            let left_value = left.value(left_row_index);
            let right_value = right.value(right_row_index);
            left_value.cmp(&right_value)
        }
    }

    fn within_range(&self, array_index: usize, row_index: usize) -> bool {
        row_index < self.arrays[array_index].len()
    }

    fn debug_value(&self, array_index: usize, row_index: usize) -> Box<dyn Debug> {
        Box::new(self.arrays[array_index].value(row_index))
    }

    fn is_valid(&self, array_index: usize, row_index: usize) -> bool {
        self.arrays[array_index].is_valid(row_index)
    }
}

#[cfg(test)]
mod tests {
    use crate::array::{ArrayRef, UInt32Array, UInt64Array};
    use crate::compute::kernels::if_op::if_primitive;
    use crate::compute::kernels::merge::{
        merge_join_indices, merge_sort_indices, MergeJoinType,
    };
    use crate::compute::{is_not_null, take};
    use crate::error::ArrowError;
    use std::sync::Arc;

    #[test]
    fn merge_sort() {
        let array_1: ArrayRef = Arc::new(UInt64Array::from(vec![1, 2, 2, 3, 5, 10, 20]));
        let array_2: ArrayRef = Arc::new(UInt64Array::from(vec![4, 8, 9, 15]));
        let array_3: ArrayRef = Arc::new(UInt64Array::from(vec![4, 7, 9, 15]));
        let vec1 = vec![array_1];
        let vec2 = vec![array_2];
        let vec3 = vec![array_3];
        let arrays = vec![vec1.as_slice(), vec2.as_slice(), vec3.as_slice()];
        let res = test_merge(arrays);

        assert_eq!(
            res.as_any().downcast_ref::<UInt64Array>().unwrap(),
            &UInt64Array::from(vec![1, 2, 2, 3, 4, 4, 5, 7, 8, 9, 9, 10, 15, 15, 20])
        )
    }

    #[test]
    fn merge_sort_with_nulls() {
        let array_1: ArrayRef = Arc::new(UInt64Array::from(vec![
            None,
            None,
            Some(1),
            Some(2),
            Some(2),
            Some(3),
            Some(5),
            Some(10),
            Some(20),
        ]));
        let array_2: ArrayRef = Arc::new(UInt64Array::from(vec![
            None,
            None,
            None,
            None,
            Some(4),
            Some(8),
            Some(9),
            Some(15),
        ]));
        let array_3: ArrayRef = Arc::new(UInt64Array::from(vec![
            None,
            Some(4),
            Some(7),
            Some(9),
            Some(15),
        ]));
        let vec1 = vec![array_1];
        let vec2 = vec![array_2];
        let vec3 = vec![array_3];
        let arrays = vec![vec1.as_slice(), vec2.as_slice(), vec3.as_slice()];
        let res = test_merge(arrays);

        assert_eq!(
            res.as_any().downcast_ref::<UInt64Array>().unwrap(),
            &UInt64Array::from(vec![
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(1),
                Some(2),
                Some(2),
                Some(3),
                Some(4),
                Some(4),
                Some(5),
                Some(7),
                Some(8),
                Some(9),
                Some(9),
                Some(10),
                Some(15),
                Some(15),
                Some(20)
            ])
        )
    }

    #[test]
    fn merge_join() {
        let array_left: ArrayRef = Arc::new(UInt64Array::from(vec![
            None,
            None,
            None,
            None,
            Some(4),
            Some(8),
            Some(9),
            Some(15),
        ]));
        let array_right: ArrayRef = Arc::new(UInt64Array::from(vec![
            None,
            Some(4),
            Some(7),
            Some(9),
            Some(15),
        ]));
        let vec_left = vec![array_left];
        let vec_right = vec![array_right];

        let result = merge_join_indices(
            vec_left.as_slice(),
            vec_right.as_slice(),
            0,
            0,
            true,
            true,
            MergeJoinType::Inner,
        )
        .unwrap();

        assert_eq!(
            result,
            (
                (8usize, false, Arc::new(UInt32Array::from(vec![4, 6, 7]))),
                (5usize, false, Arc::new(UInt32Array::from(vec![1, 3, 4])))
            )
        )
    }

    #[test]
    fn merge_left_join() {
        let array_left: ArrayRef = Arc::new(UInt64Array::from(vec![
            None,
            None,
            None,
            None,
            Some(4),
            Some(8),
            Some(9),
            Some(15),
            Some(15),
        ]));
        let array_right: ArrayRef = Arc::new(UInt64Array::from(vec![
            None,
            Some(4),
            Some(4),
            Some(7),
            Some(9),
            Some(15),
        ]));
        let vec_left = vec![array_left];
        let vec_right = vec![array_right];

        let result = merge_join_indices(
            vec_left.as_slice(),
            vec_right.as_slice(),
            0,
            0,
            true,
            true,
            MergeJoinType::Left,
        )
        .unwrap();

        assert_eq!(
            result,
            (
                (
                    9usize,
                    false,
                    Arc::new(UInt32Array::from(vec![
                        Some(0),
                        Some(1),
                        Some(2),
                        Some(3),
                        Some(4),
                        Some(4),
                        Some(5),
                        Some(6),
                        Some(7),
                        Some(8)
                    ]))
                ),
                (
                    6usize,
                    false,
                    Arc::new(UInt32Array::from(vec![
                        None,
                        None,
                        None,
                        None,
                        Some(1),
                        Some(2),
                        None,
                        Some(4),
                        Some(5),
                        Some(5)
                    ]))
                )
            )
        )
    }

    #[test]
    fn merge_left_join_empty() {
        let array_left: ArrayRef = Arc::new(UInt64Array::from(vec![
            None,
            None,
            None,
            None,
            Some(4),
            Some(8),
            Some(9),
            Some(15),
            Some(15),
        ]));
        let array_right: ArrayRef = Arc::new(UInt64Array::from(Vec::<u64>::new()));
        let vec_left = vec![array_left];
        let vec_right = vec![array_right];

        let result = merge_join_indices(
            vec_left.as_slice(),
            vec_right.as_slice(),
            0,
            0,
            true,
            true,
            MergeJoinType::Left,
        )
        .unwrap();

        assert_eq!(
            result,
            (
                (
                    9usize,
                    false,
                    Arc::new(UInt32Array::from(vec![
                        Some(0),
                        Some(1),
                        Some(2),
                        Some(3),
                        Some(4),
                        Some(5),
                        Some(6),
                        Some(7),
                        Some(8)
                    ]))
                ),
                (
                    0,
                    false,
                    Arc::new(UInt32Array::from(vec![
                        None, None, None, None, None, None, None, None, None,
                    ]))
                )
            )
        )
    }

    #[test]
    fn single_array() {
        let array_1: ArrayRef = Arc::new(UInt64Array::from(vec![1, 2, 2, 3, 5, 10, 20]));
        let vec1 = vec![array_1];
        let arrays = vec![vec1.as_slice()];
        let res = test_merge(arrays);

        assert_eq!(
            res.as_any().downcast_ref::<UInt64Array>().unwrap(),
            &UInt64Array::from(vec![1, 2, 2, 3, 5, 10, 20])
        )
    }

    fn test_merge(arrays: Vec<&[ArrayRef]>) -> ArrayRef {
        let result = merge_sort_indices(
            arrays.clone(),
            (0..arrays.len()).map(|_| 0).collect(),
            (0..arrays.len()).map(|_| true).collect(),
        )
        .unwrap();

        let mut array_values = result
            .iter()
            .enumerate()
            .map(|(i, (_, a))| take(arrays[i][0].as_ref(), a, None))
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter();
        let first = Ok(array_values.next().unwrap());
        array_values
            .fold(first, |res, b| {
                res.and_then(|a| -> Result<ArrayRef, ArrowError> {
                    Ok(Arc::new(if_primitive(
                        &is_not_null(a.as_any().downcast_ref::<UInt64Array>().unwrap())?,
                        a.as_any().downcast_ref::<UInt64Array>().unwrap(),
                        b.as_any().downcast_ref::<UInt64Array>().unwrap(),
                    )?))
                })
            })
            .unwrap()
    }
}

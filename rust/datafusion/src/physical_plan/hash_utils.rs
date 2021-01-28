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

//! Functionality used both on logical and physical plans

use crate::error::{DataFusionError, Result};
use crate::logical_plan::{DFField, DFSchema};
use std::collections::HashSet;

/// All valid types of joins.
#[derive(Clone, Copy, Debug)]
pub enum JoinType {
    /// Inner join
    Inner,
    /// Left
    Left,
    /// Right
    Right,
}

/// The on clause of the join, as vector of (left, right) columns.
pub type JoinOn = [(String, String)];

/// Checks whether the schemas "left" and "right" and columns "on" represent a valid join.
/// They are valid whenever their columns' intersection equals the set `on`
pub fn check_join_is_valid(left: &DFSchema, right: &DFSchema, on: &JoinOn) -> Result<()> {
    if on.is_empty() {
        return Err(DataFusionError::Plan(
            "The 'on' clause of a join cannot be empty".to_string(),
        ));
    }
    let on_left = &on.iter().map(|on| on.0.to_string()).collect::<HashSet<_>>();
    let left_missing = on_left
        .iter()
        .filter(|f| left.lookup_required_field_index(f).is_err())
        .collect::<HashSet<_>>();

    let on_right = &on.iter().map(|on| on.1.to_string()).collect::<HashSet<_>>();
    let right_missing = on_right
        .iter()
        .filter(|f| right.lookup_required_field_index(f).is_err())
        .collect::<HashSet<_>>();

    if !left_missing.is_empty() | !right_missing.is_empty() {
        return Err(DataFusionError::Plan(format!(
            "The left or right side of the join does not have all columns on \"on\": \nMissing on the left: {:?}\nMissing on the right: {:?}",
            left_missing,
            right_missing,
        )));
    };

    let remaining = right
        .fields()
        .iter()
        .map(|f| f.qualified_name())
        .collect::<HashSet<_>>()
        .difference(on_right)
        .cloned()
        .collect::<HashSet<String>>();

    let left_fields = left
        .fields()
        .iter()
        .map(|f| f.qualified_name())
        .collect::<HashSet<_>>();
    let collisions = left_fields.intersection(&remaining).collect::<HashSet<_>>();

    if !collisions.is_empty() {
        return Err(DataFusionError::Plan(format!(
            "The left schema and the right schema have the following columns with the same name without being on the ON statement: {:?}. Consider aliasing them.",
            collisions,
        )));
    };

    Ok(())
}

/// Creates a schema for a join operation.
/// The fields from the left side are first
pub fn build_join_schema(
    left: &DFSchema,
    right: &DFSchema,
    on: &JoinOn,
    join_type: &JoinType,
) -> Result<DFSchema> {
    let fields: Vec<DFField> = match join_type {
        JoinType::Inner | JoinType::Left => {
            // remove right-side join keys if they have the same names as the left-side
            let duplicate_keys = &on
                .iter()
                .filter(|(l, r)| l == r)
                .map(|on| on.1.to_string())
                .collect::<HashSet<_>>();

            let left_fields = left.fields().iter();

            let right_fields = right
                .fields()
                .iter()
                .filter(|f| !duplicate_keys.contains(f.name()));

            // left then right
            left_fields.chain(right_fields).cloned().collect()
        }
        JoinType::Right => {
            // remove left-side join keys if they have the same names as the right-side
            let duplicate_keys = &on
                .iter()
                .filter(|(l, r)| l == r)
                .map(|on| on.1.to_string())
                .collect::<HashSet<_>>();

            let left_fields = left
                .fields()
                .iter()
                .filter(|f| !duplicate_keys.contains(f.name()));

            let right_fields = right.fields().iter();

            // left then right
            left_fields.chain(right_fields).cloned().collect()
        }
    };
    DFSchema::new(fields)
}

#[cfg(test)]
mod tests {

    use super::*;
    use arrow::datatypes::DataType;

    fn check(left: &[&str], right: &[&str], on: &[(&str, &str)]) -> Result<()> {
        let left = DFSchema::new(
            left.iter()
                .map(|x| DFField::new(None, x, DataType::Utf8, false))
                .collect::<Vec<_>>(),
        )?;
        let right = DFSchema::new(
            right
                .iter()
                .map(|x| DFField::new(None, x, DataType::Utf8, false))
                .collect::<Vec<_>>(),
        )?;
        let on: Vec<_> = on
            .iter()
            .map(|(l, r)| (l.to_string(), r.to_string()))
            .collect();
        check_join_is_valid(&left, &right, &on)
    }

    #[test]
    fn check_valid() -> Result<()> {
        let left = vec!["a", "b1"];
        let right = vec!["a", "b2"];
        let on = &[("a", "a")];

        check(&left, &right, on)?;
        Ok(())
    }

    #[test]
    fn check_not_in_right() {
        let left = vec!["a", "b"];
        let right = vec!["b"];
        let on = &[("a", "a")];

        assert!(check(&left, &right, on).is_err());
    }

    #[test]
    fn check_not_in_left() {
        let left = vec!["b"];
        let right = vec!["a"];
        let on = &[("a", "a")];

        assert!(check(&left, &right, on).is_err());
    }

    #[test]
    fn check_collision() {
        // column "a" would appear both in left and right
        let left = vec!["a", "c"];
        let right = vec!["a", "b"];
        let on = &[("a", "b")];

        assert!(check(&left, &right, on).is_err());
    }

    #[test]
    fn check_in_right() {
        let left = vec!["a", "c"];
        let right = vec!["b"];
        let on = &[("a", "b")];

        assert!(check(&left, &right, on).is_ok());
    }
}

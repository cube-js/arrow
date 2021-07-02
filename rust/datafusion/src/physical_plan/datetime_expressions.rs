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

//! DateTime expressions
use std::sync::Arc;

use super::ColumnarValue;
use crate::{
    error::{DataFusionError, Result},
    scalar::{ScalarType, ScalarValue},
};
use arrow::array::{ArrayData, StringArray};
use arrow::buffer::Buffer;
use arrow::datatypes::ToByteSlice;
use arrow::{
    array::{Array, ArrayRef, GenericStringArray, PrimitiveArray, StringOffsetSizeTrait},
    datatypes::{ArrowPrimitiveType, DataType, TimestampNanosecondType},
};
use arrow::{
    array::{
        Date32Array, Date64Array, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray,
    },
    compute::kernels::temporal,
    datatypes::TimeUnit,
    temporal_conversions::timestamp_ns_to_datetime,
};
use chrono::format::Fixed::{Nanosecond as FixedNanosecond, TimezoneOffsetColon};
use chrono::format::Item::{Fixed, Literal, Numeric};
use chrono::format::Numeric::Nanosecond;
use chrono::format::Pad::Zero;
use chrono::format::{Item, Parsed};
use chrono::prelude::*;
use chrono::Duration;

#[inline]
/// Accepts a string in RFC3339 / ISO8601 standard format and some
/// variants and converts it to a nanosecond precision timestamp.
///
/// Implements the `to_timestamp` function to convert a string to a
/// timestamp, following the model of spark SQL’s to_`timestamp`.
///
/// In addition to RFC3339 / ISO8601 standard timestamps, it also
/// accepts strings that use a space ` ` to separate the date and time
/// as well as strings that have no explicit timezone offset.
///
/// Examples of accepted inputs:
/// * `1997-01-31T09:26:56.123Z`        # RCF3339
/// * `1997-01-31T09:26:56.123-05:00`   # RCF3339
/// * `1997-01-31 09:26:56.123-05:00`   # close to RCF3339 but with a space rather than T
/// * `1997-01-31T09:26:56.123`         # close to RCF3339 but no timezone offset specified
/// * `1997-01-31 09:26:56.123`         # close to RCF3339 but uses a space and no timezone offset
/// * `1997-01-31 09:26:56`             # close to RCF3339, no fractional seconds
//
/// Internally, this function uses the `chrono` library for the
/// datetime parsing
///
/// We hope to extend this function in the future with a second
/// parameter to specifying the format string.
///
/// ## Timestamp Precision
///
/// DataFusion uses the maximum precision timestamps supported by
/// Arrow (nanoseconds stored as a 64-bit integer) timestamps. This
/// means the range of dates that timestamps can represent is ~1677 AD
/// to 2262 AM
///
///
/// ## Timezone / Offset Handling
///
/// By using the Arrow format, DataFusion inherits Arrow’s handling of
/// timestamp values. Specifically, the stored numerical values of
/// timestamps are stored compared to offset UTC.
///
/// This function intertprets strings without an explicit time zone as
/// timestamps with offsets of the local time on the machine that ran
/// the datafusion query
///
/// For example, `1997-01-31 09:26:56.123Z` is interpreted as UTC, as
/// it has an explicit timezone specifier (“Z” for Zulu/UTC)
///
/// `1997-01-31T09:26:56.123` is interpreted as a local timestamp in
/// the timezone of the machine that ran DataFusion. For example, if
/// the system timezone is set to Americas/New_York (UTC-5) the
/// timestamp will be interpreted as though it were
/// `1997-01-31T09:26:56.123-05:00`
pub fn string_to_timestamp_nanos(s: &str) -> Result<i64> {
    // Fast path:  RFC3339 timestamp (with a T)
    // Example: 2020-09-08T13:42:29.190855Z
    if let Ok(ts) = DateTime::parse_from_rfc3339(s) {
        return Ok(ts.timestamp_nanos());
    }

    // Implement quasi-RFC3339 support by trying to parse the
    // timestamp with various other format specifiers to to support
    // separating the date and time with a space ' ' rather than 'T' to be
    // (more) compatible with Apache Spark SQL

    // We parse the date and time prefix first to share work between all the different formats.
    let mut rest = s;
    let mut p;
    let separator_is_space;
    match try_parse_prefix(&mut rest) {
        Some(ParsedPrefix {
            result,
            separator_is_space: s,
        }) => {
            p = result;
            separator_is_space = s;
        }
        None => {
            return Err(DataFusionError::Execution(format!(
                "Error parsing '{}' as timestamp",
                s
            )));
        }
    }

    if separator_is_space {
        // timezone offset, using ' ' as a separator
        // Example: 2020-09-08 13:42:29.190855-05:00
        // Full format string: "%Y-%m-%d %H:%M:%S%.f%:z".
        const FORMAT1: [Item; 2] = [Fixed(FixedNanosecond), Fixed(TimezoneOffsetColon)];
        if let Ok(ts) = chrono::format::parse(&mut p, rest, FORMAT1.iter())
            .and_then(|()| p.to_datetime())
        {
            return Ok(ts.timestamp_nanos());
        }

        // with an explicit Z, using ' ' as a separator
        // Example: 2020-09-08 13:42:29Z
        // Full format string: "%Y-%m-%d %H:%M:%S%.fZ".
        const FORMAT2: [Item; 2] = [Fixed(FixedNanosecond), Literal("Z")];
        if let Ok(ts) = chrono::format::parse(&mut p, rest, FORMAT2.iter())
            .and_then(|()| p.to_datetime_with_timezone(&Utc))
        {
            return Ok(ts.timestamp_nanos());
        }

        // without a timezone specifier as a local time, using ' ' as a separator
        // Example: 2020-09-08 13:42:29.190855
        const FORMAT5: [Item; 2] = [Literal("."), Numeric(Nanosecond, Zero)];
        // Full format string: "%Y-%m-%d %H:%M:%S.%f".
        if let Ok(ts) = chrono::format::parse(&mut p, rest, FORMAT5.iter())
            .and_then(|()| p.to_naive_datetime_with_offset(0))
        {
            return naive_datetime_to_timestamp(s, ts);
        }

        // without a timezone specifier as a local time, using ' ' as a
        // separator, no fractional seconds
        // Example: 2020-09-08 13:42:29
        // Full format string: "%Y-%m-%d %H:%M:%S".
        if rest.is_empty() {
            if let Ok(ts) = p.to_naive_datetime_with_offset(0) {
                return naive_datetime_to_timestamp(s, ts);
            }
        }
    }

    // Support timestamps without an explicit timezone offset, again
    // to be compatible with what Apache Spark SQL does.
    if !separator_is_space
    /* i.e. separator == b'T' */
    {
        // without a timezone specifier as a local time, using T as a separator
        // Example: 2020-09-08T13:42:29.190855
        // Full format string: "%Y-%m-%dT%H:%M:%S.%f".
        const FORMAT3: [Item; 2] = [Literal("."), Numeric(Nanosecond, Zero)];
        if let Ok(ts) = chrono::format::parse(&mut p, rest, FORMAT3.iter())
            .and_then(|()| p.to_naive_datetime_with_offset(0))
        {
            return naive_datetime_to_timestamp(s, ts);
        }

        // without a timezone specifier as a local time, using T as a
        // separator, no fractional seconds
        // Example: 2020-09-08T13:42:29
        // Full format string: "%Y-%m-%dT%H:%M:%S".
        if rest.is_empty() {
            if let Ok(ts) = p.to_naive_datetime_with_offset(0) {
                return naive_datetime_to_timestamp(s, ts);
            }
        }
    }

    // Note we don't pass along the error message from the underlying
    // chrono parsing because we tried several different format
    // strings and we don't know which the user was trying to
    // match. Ths any of the specific error messages is likely to be
    // be more confusing than helpful
    Err(DataFusionError::Execution(format!(
        "Error parsing '{}' as timestamp",
        s
    )))
}

#[must_use]
fn try_parse_num(s: &mut &[u8]) -> Option<i64> {
    if s.is_empty() {
        return None;
    }

    let mut i;
    if s[0] == b'-' {
        i = 1
    } else {
        i = 0;
    }

    while i < s.len() && b'0' <= s[i] && s[i] <= b'9' {
        i += 1
    }

    let res = unsafe { std::str::from_utf8_unchecked(&s[0..i]) }
        .parse()
        .ok();
    *s = &s[i..];
    return res;
}

#[must_use]
fn try_consume(s: &mut &[u8], c: u8) -> Option<()> {
    if s.is_empty() || s[0] != c {
        return None;
    }
    *s = &s[1..];
    Some(())
}

struct ParsedPrefix {
    result: Parsed,
    separator_is_space: bool, // When false, the separator is 'T'.
}

/// Parses YYYY-MM-DD(T| )HH:MM:SS.
fn try_parse_prefix(s: &mut &str) -> Option<ParsedPrefix> {
    let mut p = Parsed::new();

    let mut rest = s.as_bytes();
    let year = try_parse_num(&mut rest)?;
    try_consume(&mut rest, b'-')?;
    let month = try_parse_num(&mut rest)?;
    try_consume(&mut rest, b'-')?;
    let day = try_parse_num(&mut rest)?;
    if rest.is_empty() {
        return None;
    }

    let separator_is_space;
    match rest[0] {
        b' ' => separator_is_space = true,
        b'T' => separator_is_space = false,
        _ => return None,
    }

    rest = &rest[1..];
    let hour = try_parse_num(&mut rest)?;
    try_consume(&mut rest, b':')?;
    let minute = try_parse_num(&mut rest)?;
    try_consume(&mut rest, b':')?;
    let second = try_parse_num(&mut rest)?;

    p.set_year(year).ok()?;
    p.set_month(month).ok()?;
    p.set_day(day).ok()?;
    p.set_hour(hour).ok()?;
    p.set_minute(minute).ok()?;
    p.set_second(second).ok()?;

    *s = unsafe { std::str::from_utf8_unchecked(rest) };
    Some(ParsedPrefix {
        result: p,
        separator_is_space,
    })
}

/// Converts the naive datetime (which has no specific timezone) to a
/// nanosecond epoch timestamp relative to UTC.
fn naive_datetime_to_timestamp(_s: &str, datetime: NaiveDateTime) -> Result<i64> {
    Ok(Utc.from_utc_datetime(&datetime).timestamp_nanos())
}

/// convert_tz SQL function
pub fn convert_tz(args: &[ArrayRef]) -> Result<ArrayRef> {
    let timestamps = &args[0]
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .ok_or_else(|| {
            DataFusionError::Execution(
                "Could not cast convert_tz timestamp input to TimestampNanosecondArray"
                    .to_string(),
            )
        })?;

    let shift = &args[1]
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            DataFusionError::Execution(
                "Could not cast convert_tz shift input to StringArray".to_string(),
            )
        })?;

    let range = 0..timestamps.len();
    let result = range
        .map(|i| {
            if timestamps.is_null(i) {
                Ok(0_i64)
            } else {
                let hour_min = shift.value(i).split(':').collect::<Vec<_>>();
                if hour_min.len() != 2 {
                    return Err(DataFusionError::Execution(format!(
                        "Can't parse timezone shift '{}'",
                        shift.value(i)
                    )));
                }
                let hour = hour_min[0].parse::<i64>().map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Can't parse hours of timezone shift '{}': {}",
                        hour_min[0], e
                    ))
                })?;
                let minute = hour_min[1].parse::<i64>().map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Can't parse minutes of timezone shift '{}': {}",
                        hour_min[1], e
                    ))
                })?;
                let shift = (hour * 60 + hour.signum() * minute) * 60 * 1_000_000_000;
                Ok(timestamps.value(i) + shift)
            }
        })
        .collect::<Result<Vec<_>>>()?;

    let data = ArrayData::new(
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        timestamps.len(),
        Some(timestamps.null_count()),
        timestamps.data().null_buffer().cloned(),
        0,
        vec![Buffer::from(result.to_byte_slice())],
        vec![],
    );

    Ok(Arc::new(TimestampNanosecondArray::from(data)))
}

// given a function `op` that maps a `&str` to a Result of an arrow native type,
// returns a `PrimitiveArray` after the application
// of the function to `args[0]`.
/// # Errors
/// This function errors iff:
/// * the number of arguments is not 1 or
/// * the first argument is not castable to a `GenericStringArray` or
/// * the function `op` errors
pub(crate) fn unary_string_to_primitive_function<'a, T, O, F>(
    args: &[&'a dyn Array],
    op: F,
    name: &str,
) -> Result<PrimitiveArray<O>>
where
    O: ArrowPrimitiveType,
    T: StringOffsetSizeTrait,
    F: Fn(&'a str) -> Result<O::Native>,
{
    if args.len() != 1 {
        return Err(DataFusionError::Internal(format!(
            "{:?} args were supplied but {} takes exactly one argument",
            args.len(),
            name,
        )));
    }

    let array = args[0]
        .as_any()
        .downcast_ref::<GenericStringArray<T>>()
        .ok_or_else(|| {
            DataFusionError::Internal("failed to downcast to string".to_string())
        })?;

    // first map is the iterator, second is for the `Option<_>`
    array.iter().map(|x| x.map(|x| op(x)).transpose()).collect()
}

// given an function that maps a `&str` to a arrow native type,
// returns a `ColumnarValue` where the function is applied to either a `ArrayRef` or `ScalarValue`
// depending on the `args`'s variant.
fn handle<'a, O, F, S>(
    args: &'a [ColumnarValue],
    op: F,
    name: &str,
) -> Result<ColumnarValue>
where
    O: ArrowPrimitiveType,
    S: ScalarType<O::Native>,
    F: Fn(&'a str) -> Result<O::Native>,
{
    match &args[0] {
        ColumnarValue::Array(a) => match a.data_type() {
            DataType::Utf8 => Ok(ColumnarValue::Array(Arc::new(
                unary_string_to_primitive_function::<i32, O, _>(&[a.as_ref()], op, name)?,
            ))),
            DataType::LargeUtf8 => Ok(ColumnarValue::Array(Arc::new(
                unary_string_to_primitive_function::<i64, O, _>(&[a.as_ref()], op, name)?,
            ))),
            other => Err(DataFusionError::Internal(format!(
                "Unsupported data type {:?} for function {}",
                other, name,
            ))),
        },
        ColumnarValue::Scalar(scalar) => match scalar {
            ScalarValue::Utf8(a) => {
                let result = a.as_ref().map(|x| (op)(x)).transpose()?;
                Ok(ColumnarValue::Scalar(S::scalar(result)))
            }
            ScalarValue::LargeUtf8(a) => {
                let result = a.as_ref().map(|x| (op)(x)).transpose()?;
                Ok(ColumnarValue::Scalar(S::scalar(result)))
            }
            other => Err(DataFusionError::Internal(format!(
                "Unsupported data type {:?} for function {}",
                other, name
            ))),
        },
    }
}

/// to_timestamp SQL function
pub fn to_timestamp(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    handle::<TimestampNanosecondType, _, TimestampNanosecondType>(
        args,
        string_to_timestamp_nanos,
        "to_timestamp",
    )
}

/// Create an implementation of `now()` that always returns the
/// specified timestamp.
///
/// The semantics of `now()` require it to return the same value
/// whenever it is called in a query. This this value is chosen during
/// planning time and bound into a closure that
pub fn make_now(
    now_ts: DateTime<Utc>,
) -> impl Fn(&[ColumnarValue]) -> Result<ColumnarValue> {
    let now_ts = Some(now_ts.timestamp_nanos());
    move |_arg| {
        Ok(ColumnarValue::Scalar(ScalarValue::TimestampNanosecond(
            now_ts,
        )))
    }
}

fn date_trunc_single(granularity: &str, value: i64) -> Result<i64> {
    let value = timestamp_ns_to_datetime(value).with_nanosecond(0);
    let value = match granularity {
        "second" => value,
        "minute" => value.and_then(|d| d.with_second(0)),
        "hour" => value
            .and_then(|d| d.with_second(0))
            .and_then(|d| d.with_minute(0)),
        "day" => value
            .and_then(|d| d.with_second(0))
            .and_then(|d| d.with_minute(0))
            .and_then(|d| d.with_hour(0)),
        "week" => value
            .and_then(|d| d.with_second(0))
            .and_then(|d| d.with_minute(0))
            .and_then(|d| d.with_hour(0))
            .map(|d| d - Duration::seconds(60 * 60 * 24 * d.weekday() as i64)),
        "month" => value
            .and_then(|d| d.with_second(0))
            .and_then(|d| d.with_minute(0))
            .and_then(|d| d.with_hour(0))
            .and_then(|d| d.with_day0(0)),
        "year" => value
            .and_then(|d| d.with_second(0))
            .and_then(|d| d.with_minute(0))
            .and_then(|d| d.with_hour(0))
            .and_then(|d| d.with_day0(0))
            .and_then(|d| d.with_month0(0)),
        unsupported => {
            return Err(DataFusionError::Execution(format!(
                "Unsupported date_trunc granularity: {}",
                unsupported
            )));
        }
    };
    // `with_x(0)` are infalible because `0` are always a valid
    Ok(value.unwrap().timestamp_nanos())
}

/// date_trunc SQL function
pub fn date_trunc(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let (granularity, array) = (&args[0], &args[1]);

    let granularity =
        if let ColumnarValue::Scalar(ScalarValue::Utf8(Some(v))) = granularity {
            v
        } else {
            return Err(DataFusionError::Execution(
                "Granularity of `date_trunc` must be non-null scalar Utf8".to_string(),
            ));
        };

    let f = |x: Option<i64>| x.map(|x| date_trunc_single(granularity, x)).transpose();

    Ok(match array {
        ColumnarValue::Scalar(scalar) => {
            if let ScalarValue::TimestampNanosecond(v) = scalar {
                ColumnarValue::Scalar(ScalarValue::TimestampNanosecond((f)(*v)?))
            } else {
                return Err(DataFusionError::Execution(
                    "array of `date_trunc` must be non-null scalar Utf8".to_string(),
                ));
            }
        }
        ColumnarValue::Array(array) => {
            let array = array
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap();
            let array = array
                .iter()
                .map(f)
                .collect::<Result<TimestampNanosecondArray>>()?;

            ColumnarValue::Array(Arc::new(array))
        }
    })
}

macro_rules! extract_date_part {
    ($ARRAY: expr, $FN:expr) => {
        match $ARRAY.data_type() {
            DataType::Date32 => {
                let array = $ARRAY.as_any().downcast_ref::<Date32Array>().unwrap();
                Ok($FN(array)?)
            }
            DataType::Date64 => {
                let array = $ARRAY.as_any().downcast_ref::<Date64Array>().unwrap();
                Ok($FN(array)?)
            }
            DataType::Timestamp(time_unit, None) => match time_unit {
                TimeUnit::Second => {
                    let array = $ARRAY
                        .as_any()
                        .downcast_ref::<TimestampSecondArray>()
                        .unwrap();
                    Ok($FN(array)?)
                }
                TimeUnit::Millisecond => {
                    let array = $ARRAY
                        .as_any()
                        .downcast_ref::<TimestampMillisecondArray>()
                        .unwrap();
                    Ok($FN(array)?)
                }
                TimeUnit::Microsecond => {
                    let array = $ARRAY
                        .as_any()
                        .downcast_ref::<TimestampMicrosecondArray>()
                        .unwrap();
                    Ok($FN(array)?)
                }
                TimeUnit::Nanosecond => {
                    let array = $ARRAY
                        .as_any()
                        .downcast_ref::<TimestampNanosecondArray>()
                        .unwrap();
                    Ok($FN(array)?)
                }
            },
            datatype => Err(DataFusionError::Internal(format!(
                "Extract does not support datatype {:?}",
                datatype
            ))),
        }
    };
}

/// DATE_PART SQL function
pub fn date_part(args: &[ColumnarValue]) -> Result<ColumnarValue> {
    if args.len() != 2 {
        return Err(DataFusionError::Execution(
            "Expected two arguments in DATE_PART".to_string(),
        ));
    }
    let (date_part, array) = (&args[0], &args[1]);

    let date_part = if let ColumnarValue::Scalar(ScalarValue::Utf8(Some(v))) = date_part {
        v
    } else {
        return Err(DataFusionError::Execution(
            "First argument of `DATE_PART` must be non-null scalar Utf8".to_string(),
        ));
    };

    let is_scalar = matches!(array, ColumnarValue::Scalar(_));

    let array = match array {
        ColumnarValue::Array(array) => array.clone(),
        ColumnarValue::Scalar(scalar) => scalar.to_array(),
    };

    let arr = match date_part.to_lowercase().as_str() {
        "hour" => extract_date_part!(array, temporal::hour),
        "year" => extract_date_part!(array, temporal::year),
        _ => Err(DataFusionError::Execution(format!(
            "Date part '{}' not supported",
            date_part
        ))),
    }?;

    Ok(if is_scalar {
        ColumnarValue::Scalar(ScalarValue::try_from_array(
            &(Arc::new(arr) as ArrayRef),
            0,
        )?)
    } else {
        ColumnarValue::Array(Arc::new(arr))
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Int64Array, StringBuilder};

    use super::*;

    #[test]
    fn to_timestamp_arrays_and_nulls() -> Result<()> {
        // ensure that arrow array implementation is wired up and handles nulls correctly

        let mut string_builder = StringBuilder::new(2);
        let mut ts_builder = TimestampNanosecondArray::builder(2);

        string_builder.append_value("2020-09-08T13:42:29.190855Z")?;
        ts_builder.append_value(1599572549190855000)?;

        string_builder.append_null()?;
        ts_builder.append_null()?;
        let expected_timestamps = &ts_builder.finish() as &dyn Array;

        let string_array =
            ColumnarValue::Array(Arc::new(string_builder.finish()) as ArrayRef);
        let parsed_timestamps = to_timestamp(&[string_array])
            .expect("that to_timestamp parsed values without error");
        if let ColumnarValue::Array(parsed_array) = parsed_timestamps {
            assert_eq!(parsed_array.len(), 2);
            assert_eq!(expected_timestamps, parsed_array.as_ref());
        } else {
            panic!("Expected a columnar array")
        }
        Ok(())
    }

    #[test]
    fn date_trunc_test() {
        let cases = vec![
            (
                "2020-09-08T13:42:29.190855Z",
                "second",
                "2020-09-08T13:42:29.000000Z",
            ),
            (
                "2020-09-08T13:42:29.190855Z",
                "minute",
                "2020-09-08T13:42:00.000000Z",
            ),
            (
                "2020-09-08T13:42:29.190855Z",
                "hour",
                "2020-09-08T13:00:00.000000Z",
            ),
            (
                "2020-09-08T13:42:29.190855Z",
                "day",
                "2020-09-08T00:00:00.000000Z",
            ),
            (
                "2020-09-08T13:42:29.190855Z",
                "week",
                "2020-09-07T00:00:00.000000Z",
            ),
            (
                "2020-09-08T13:42:29.190855Z",
                "month",
                "2020-09-01T00:00:00.000000Z",
            ),
            (
                "2020-09-08T13:42:29.190855Z",
                "year",
                "2020-01-01T00:00:00.000000Z",
            ),
            (
                "2021-01-01T13:42:29.190855Z",
                "week",
                "2020-12-28T00:00:00.000000Z",
            ),
            (
                "2020-01-01T13:42:29.190855Z",
                "week",
                "2019-12-30T00:00:00.000000Z",
            ),
        ];

        cases.iter().for_each(|(original, granularity, expected)| {
            let original = string_to_timestamp_nanos(original).unwrap();
            let expected = string_to_timestamp_nanos(expected).unwrap();
            let result = date_trunc_single(granularity, original).unwrap();
            assert_eq!(result, expected);
        });
    }

    #[test]
    fn to_timestamp_invalid_input_type() -> Result<()> {
        // pass the wrong type of input array to to_timestamp and test
        // that we get an error.

        let mut builder = Int64Array::builder(1);
        builder.append_value(1)?;
        let int64array = ColumnarValue::Array(Arc::new(builder.finish()));

        let expected_err =
            "Internal error: Unsupported data type Int64 for function to_timestamp";
        match to_timestamp(&[int64array]) {
            Ok(_) => panic!("Expected error but got success"),
            Err(e) => {
                assert!(
                    e.to_string().contains(expected_err),
                    "Can not find expected error '{}'. Actual error '{}'",
                    expected_err,
                    e
                );
            }
        }
        Ok(())
    }
}

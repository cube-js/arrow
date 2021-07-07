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

use futures::Future;
use tokio::task::JoinHandle;
use tracing_futures::Instrument;

/// Calls [tokio::spawn] and additionally enables tracing of the spawned task as part of the current
/// computation. This is CubeStore approach to tracing, so all code must use this function instead
/// of replace [tokio::spawn].
pub fn spawn<T>(task: T) -> JoinHandle<T::Output>
where
    T: Future + Send + 'static,
    T::Output: Send + 'static,
{
    if let Some(s) = new_subtask_span() {
        tokio::spawn(async move {
            let _p = s.parent; // ensure parent stays alive.
            task.instrument(s.child).await
        })
    } else {
        tokio::spawn(task)
    }
}

/// Propagates current span to blocking operation. See [spawn] for details.
pub fn spawn_blocking<F, R>(f: F) -> JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    if let Some(s) = new_subtask_span() {
        tokio::task::spawn_blocking(move || {
            let _p = s.parent; // ensure parent stays alive.
            s.child.in_scope(f)
        })
    } else {
        tokio::task::spawn_blocking(f)
    }
}

struct SpawnSpans {
    parent: tracing::Span,
    child: tracing::Span,
}

fn new_subtask_span() -> Option<SpawnSpans> {
    let parent = tracing::Span::current();
    if parent.is_disabled() {
        return None;
    }
    // TODO: ensure this is always enabled.
    let child = tracing::info_span!(parent: &parent, "subtask");
    Some(SpawnSpans { parent, child })
}

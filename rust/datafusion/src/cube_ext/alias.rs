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

use crate::error::Result;
use crate::execution::context::ExecutionContextState;
use crate::logical_plan::{DFSchemaRef, Expr, LogicalPlan, UserDefinedLogicalNode};
use crate::physical_plan::alias::AliasedSchemaExec;
use crate::physical_plan::planner::ExtensionPlanner;
use crate::physical_plan::ExecutionPlan;
use smallvec::alloc::fmt::Formatter;
use std::any::Any;
use std::sync::Arc;

#[derive(Debug)]
pub struct LogicalAlias {
    pub input: LogicalPlan,
    pub alias: String,
    pub schema: DFSchemaRef,
}

impl LogicalAlias {
    pub fn new(input: LogicalPlan, alias: String) -> Result<LogicalAlias> {
        let schema = Arc::new(input.schema().alias(Some(&alias))?);
        Ok(LogicalAlias {
            input,
            alias,
            schema,
        })
    }
}

impl UserDefinedLogicalNode for LogicalAlias {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        Vec::new()
    }

    fn fmt_for_explain(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "Alias as {}", self.alias)
    }

    fn from_template(
        &self,
        exprs: &[Expr],
        inputs: &[LogicalPlan],
    ) -> Arc<dyn UserDefinedLogicalNode + Send + Sync> {
        assert_eq!(exprs.len(), 0);
        assert_eq!(inputs.len(), 1);

        Arc::new(LogicalAlias {
            input: inputs[0].clone(),
            alias: self.alias.clone(),
            schema: self.schema.clone(),
        })
    }
}

pub struct LogicalAliasPlanner;
impl ExtensionPlanner for LogicalAliasPlanner {
    fn plan_extension(
        &self,
        node: &dyn UserDefinedLogicalNode,
        inputs: &[Arc<dyn ExecutionPlan>],
        _: &ExecutionContextState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let node = match node.as_any().downcast_ref::<LogicalAlias>() {
            None => return Ok(None),
            Some(n) => n,
        };

        assert_eq!(inputs.len(), 1);
        log::info!("Input is {:#?}", inputs[0]);
        Ok(Some(AliasedSchemaExec::wrap(
            Some(node.alias.clone()),
            inputs[0].clone(),
        )))
    }
}

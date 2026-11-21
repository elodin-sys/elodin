//! Moves calls that do not depend on one another next to each other at the top level of a
//! function body, so `lower` can run each run of them as one group on [`crate::pool`].

use std::collections::{BTreeSet, HashMap, HashSet};
use std::ops::Range;

use crate::ir::{FuncDef, InstrResult, Instruction, Module, ValueId};

/// Groups the calls of every function `can_group` accepts. Returns, for each function with a
/// group, the body positions of its runs of adjacent independent calls.
pub fn group_module_calls(
    module: &mut Module,
    can_group: impl Fn(&FuncDef) -> bool,
) -> HashMap<String, Vec<Range<usize>>> {
    let mut groups = HashMap::new();
    for func in &mut module.functions {
        if !can_group(func) {
            continue;
        }
        let runs = group_calls(&mut func.body);
        if !runs.is_empty() {
            groups.insert(func.name.clone(), runs);
        }
    }
    groups
}

/// A call's level is the number of calls on the longest dependency chain that feeds it, so calls
/// of one level never depend on each other. Every call moves up next to the first call of its
/// level, preceded by the instructions it needs from in between; nothing else moves. Returns the
/// runs of two or more adjacent calls of one level.
pub fn group_calls(body: &mut Vec<InstrResult>) -> Vec<Range<usize>> {
    let levels = call_levels(body);
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for level in levels.values() {
        *counts.entry(*level).or_default() += 1;
    }
    let mut shared: Vec<usize> = counts
        .into_iter()
        .filter(|&(_, count)| count > 1)
        .map(|(level, _)| level)
        .collect();
    shared.sort_unstable();
    for level in shared {
        hoist(body, level, &levels);
    }
    runs(body, &levels)
}

fn call_id(ir: &InstrResult) -> Option<ValueId> {
    match ir.instr {
        Instruction::Call { .. } => ir.values.first().map(|(vid, _)| *vid),
        _ => None,
    }
}

/// Values defined outside `ir` that it reads, including from its nested regions.
fn uses(ir: &InstrResult) -> Vec<ValueId> {
    crate::lower::collect_captured_vids(std::slice::from_ref(ir))
}

fn call_levels(body: &[InstrResult]) -> HashMap<ValueId, usize> {
    let mut value_levels: HashMap<ValueId, usize> = HashMap::new();
    let mut levels = HashMap::new();
    for ir in body {
        let level = uses(ir)
            .iter()
            .filter_map(|vid| value_levels.get(vid))
            .max()
            .copied()
            .unwrap_or(0);
        let call = call_id(ir);
        if let Some(id) = call {
            levels.insert(id, level);
        }
        for (vid, _) in &ir.values {
            value_levels.insert(*vid, level + usize::from(call.is_some()));
        }
    }
    levels
}

fn level_of(ir: &InstrResult, levels: &HashMap<ValueId, usize>) -> Option<usize> {
    call_id(ir).and_then(|id| levels.get(&id).copied())
}

fn hoist(body: &mut Vec<InstrResult>, level: usize, levels: &HashMap<ValueId, usize>) {
    let members: Vec<usize> = (0..body.len())
        .filter(|&i| level_of(&body[i], levels) == Some(level))
        .collect();
    let Some((&first, rest)) = members.split_first() else {
        return;
    };
    let defined_at: HashMap<ValueId, usize> = body
        .iter()
        .enumerate()
        .flat_map(|(i, ir)| ir.values.iter().map(move |(vid, _)| (*vid, i)))
        .collect();
    let moved: HashSet<usize> = rest.iter().copied().collect();
    let mut needed = BTreeSet::new();
    for &member in rest {
        let mut pending = uses(&body[member]);
        while let Some(vid) = pending.pop() {
            if let Some(&at) = defined_at.get(&vid)
                && at > first
                && !moved.contains(&at)
                && needed.insert(at)
            {
                pending.extend(uses(&body[at]));
            }
        }
    }
    let mut slots: Vec<Option<InstrResult>> = std::mem::take(body).into_iter().map(Some).collect();
    let mut take = |i: usize| slots[i].take().expect("each instruction moves once");
    let mut reordered: Vec<InstrResult> = (0..first).map(&mut take).collect();
    reordered.extend(needed.iter().map(|&i| take(i)));
    reordered.push(take(first));
    reordered.extend(rest.iter().map(|&i| take(i)));
    reordered.extend(slots.into_iter().flatten());
    *body = reordered;
}

fn runs(body: &[InstrResult], levels: &HashMap<ValueId, usize>) -> Vec<Range<usize>> {
    let mut runs = Vec::new();
    let mut start = 0;
    while start < body.len() {
        let Some(level) = level_of(&body[start], levels) else {
            start += 1;
            continue;
        };
        let end = (start + 1..body.len())
            .find(|&i| level_of(&body[i], levels) != Some(level))
            .unwrap_or(body.len());
        if end - start > 1 {
            runs.push(start..end);
        }
        start = end;
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ElementType, TensorType};

    fn instr(result: u32, instr: Instruction) -> InstrResult {
        InstrResult {
            values: vec![(ValueId(result), TensorType::scalar(ElementType::F64))],
            instr,
        }
    }

    fn call(result: u32, args: &[u32]) -> InstrResult {
        instr(
            result,
            Instruction::Call {
                callee: "f".into(),
                args: args.iter().map(|&a| ValueId(a)).collect(),
            },
        )
    }

    fn negate(result: u32, operand: u32) -> InstrResult {
        instr(
            result,
            Instruction::Negate {
                operand: ValueId(operand),
            },
        )
    }

    fn results(body: &[InstrResult]) -> Vec<u32> {
        body.iter().map(|ir| ir.values[0].0.0).collect()
    }

    #[test]
    fn independent_calls_move_together_with_their_operands() {
        // 1 = -0; 2 = f(1); 3 = -0; 4 = f(3); 5 = f(2, 4); 6 = -5; 7 = f(6)
        let mut body = vec![
            negate(1, 0),
            call(2, &[1]),
            negate(3, 0),
            call(4, &[3]),
            call(5, &[2, 4]),
            negate(6, 5),
            call(7, &[6]),
        ];
        let groups = group_calls(&mut body);
        assert_eq!(results(&body), [1, 3, 2, 4, 5, 6, 7]);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], 2..4);
    }

    #[test]
    fn dependent_calls_stay_apart() {
        let mut body = vec![call(1, &[0]), negate(2, 1), call(3, &[2])];
        assert!(group_calls(&mut body).is_empty());
        assert_eq!(results(&body), [1, 2, 3]);
    }
}

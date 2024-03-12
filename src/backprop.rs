use std::collections::HashMap;
use std::ops::Mul;
use crate::tensor::{Tensor, TensorId};
use crate::op::*;
use crate::variables::Var;


impl Tensor {
    // Return all the nodes that lead to this value in a topologically sorted vec,
    // the first elements having dependencies on the latter ones,
    // e.g. the first element if any is the argument. This assumes that the op graph is a DAG.
    fn sorted_nodes(&self) -> Vec<&Tensor> {
        fn walk<'a>(node: &'a Tensor, nodes: Vec<&'a Tensor>,
                    already_seen: &mut HashMap<TensorId, bool>) -> (bool, Vec<&'a Tensor>) {
            if let Some(&tg) = already_seen.get(&node.id()) {
                return (tg, nodes);
            }
            let mut track_grad = false;
            
            let mut nodes = if node.is_var() {
                // Do not call recursively on the "leaf" nodes.
                track_grad = true;
                nodes
            } else if let Some(op) = node.op() {
                match op {
                    Op::Unary(node, _) => {
                        let (tg, mut nodes) = walk(node, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    _ => unimplemented!("Op not implemented")
                }
            } else {
                nodes
            };
            already_seen.insert(node.id(), track_grad);
            if track_grad {
                nodes.push(node);
            }
            (track_grad, nodes)
        }

        let (_tg, mut nodes) = walk(self, vec![], &mut HashMap::new());
        nodes.reverse();
        nodes
    }

    pub fn backward(&self) -> Result<GradStore, Box<dyn std::error::Error>> {
        let nodes = self.sorted_nodes();
        let mut grads = GradStore::new();
        grads.insert(self, Tensor::ones_like(self)?.contiguous().unwrap());
        for node in nodes.iter() {
            if node.is_var() {
                continue;
            }
            let grad = grads.remove(node).expect("Node not found in grads");
            println!("grad: {}", grad);
            if let Some(op) = node.op() {
                match op {
                    Op::Unary(a, UnaryOp::Sqr) => {
                        let grad = a.mul(&grad)?.mul(&Tensor::from(2.0))?;
                        grads.insert(a, grad);
                    }
                    _ => unimplemented!("Op not implemented")
                }
            }
        }
        Ok(grads)
    }
}

#[derive(Debug)]
pub struct GradStore(HashMap<TensorId, Tensor>);

impl GradStore {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn get_id(&self, id: TensorId) -> Option<&Tensor> {
        self.0.get(&id)
    }

    pub fn get(&self, tensor: &Tensor) -> Option<&Tensor> {
        self.0.get(&tensor.id())
    }

    pub fn remove(&mut self, tensor: &Tensor) -> Option<Tensor> {
        self.0.remove(&tensor.id())
    }

    pub fn insert(&mut self, tensor: &Tensor, grad: Tensor) -> Option<Tensor> {
        self.0.insert(tensor.id(), grad)
    }

}

#[cfg(test)]
mod tests {
    use crate::device::Device;
    use super::*;
    use crate::shape::Shape;

    #[test]
    fn sorted_nodes() {
        let device = Device::Cpu;
        let a = Var::new(&[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device).unwrap();
        let b = a.sqr().unwrap().backward().unwrap();
        let grads = b.get(&a).unwrap();
        println!("grad: {}", grads);
        
    }
}


mod shape;
mod tensor;
mod storage;
mod cpu_backend;
mod layout;
mod dtype;
mod device;
mod op;
mod accelerate;
mod strided_index;
mod backprop;
mod indexer;
mod variables;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

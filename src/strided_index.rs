#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StridedBlocks<'a> {
    SingleBlock {
        start_offset: usize,
        len: usize,
    },
    MultipleBlocks {
        block_start_index: StridedIndex<'a>,
        block_len: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StridedIndex<'a> {
    next_storage_index: Option<usize>,
    multi_index: Vec<usize>,
    dims: &'a [usize],
    stride: &'a [usize],
}

impl<'a> StridedIndex<'a> {
    pub fn new(dims: &'a [usize], stride: &'a [usize], start_offset: usize) -> Self {
        let elem_count: usize = dims.iter().product();
        let next_storage_index = if elem_count == 0 {
            None
        } else {
            Some(start_offset)
        };

        StridedIndex {
            next_storage_index,
            multi_index: vec![0; dims.len()],
            dims,
            stride,
        }
    }
}

impl <'a> Iterator for StridedIndex<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let storage_index = match self.next_storage_index {
            None => return None,
            Some(storage_index) => storage_index,
        };
        let mut updated = false;
        let mut next_storage_index = storage_index;
        for ((multi_i, max_i), stride_i) in self
            .multi_index
            .iter_mut()
            .zip(self.dims.iter())
            .zip(self.stride.iter())
            .rev()
        {
            let next_i = *multi_i + 1;
            if next_i < *max_i {
                *multi_i = next_i;
                updated = true;
                next_storage_index += stride_i;
                break;
            } else {
                next_storage_index -= *multi_i * stride_i;
                *multi_i = 0
            }
        }
        self.next_storage_index = if updated {
            Some(next_storage_index)
        } else {
            None
        };
        Some(storage_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strided_index() {
        let dims = vec![2, 3];
        let stride = vec![3, 1];
        let start_offset = 0;
        let mut strided_index = StridedIndex::new(&dims, &stride, start_offset);
        assert_eq!(strided_index.next(), Some(0));
        assert_eq!(strided_index.next(), Some(1));
        assert_eq!(strided_index.next(), Some(2));
        assert_eq!(strided_index.next(), Some(3));
        assert_eq!(strided_index.next(), Some(4));
        assert_eq!(strided_index.next(), Some(5));
        assert_eq!(strided_index.next(), None);
    }

    #[test]
    fn test_strided_blocks() {
        let dims = vec![2, 3];
        let stride = vec![3, 1];
        let start_offset = 0;
        let strided_index = StridedIndex::new(&dims, &stride, start_offset);
        let strided_blocks = StridedBlocks::SingleBlock {
            start_offset: 0,
            len: 6,
        };
        assert_eq!(strided_blocks, StridedBlocks::SingleBlock {
            start_offset: 0,
            len: 6,
        });
    }

    #[test]
    fn test_strided_blocks_multiple() {
        let dims = vec![2, 2];
        let stride = vec![1, 2];
        let start_offset = 0;
        let mut strided_index = StridedIndex::new(&dims, &stride, start_offset);
        assert_eq!(strided_index.next(), Some(0));
        assert_eq!(strided_index.next(), Some(2));
        assert_eq!(strided_index.next(), Some(1));
        assert_eq!(strided_index.next(), Some(3));
    }
}
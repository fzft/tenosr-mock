#[derive(Debug)]
pub enum StridedBlocks {
    SingleBlock {
        start_offset: usize,
        len: usize,
    },
    MultipleBlocks {
        block_start_index: usize,
        block_len: usize,
    },
}
use crate::cpu_backend::CpuStorage;
use crate::device::Device;
use crate::dtype::Dtype;
use crate::layout::Layout;
use crate::op::UnaryOpT;


#[derive(Debug)]
pub enum Storage {
   Cpu(CpuStorage)
}

impl Storage {
   pub fn device(&self) -> Device {
       match self {
           Storage::Cpu(_) => Device::Cpu,
       }
   }


   pub fn dtype(&self) -> Dtype {
       match self {
           Self::Cpu(storage) => storage.dtype(),
       }
   }

    pub fn copy_strided_src(&self,
                            dst: &mut Self,
                            dst_offset: usize,
                            src_l: &Layout
    ) -> Result<(), Box<dyn std::error::Error>> {
        match (self, dst) {
            (Self::Cpu(src), Self::Cpu(dst)) => {
                src.copy_strided_src(dst, dst_offset, src_l);
                Ok(())
            }
        }
    }

    pub fn matmul(&self,
                  rhs: &Self,
                  bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        match (self, rhs) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                let storage = lhs.matmul(rhs, bmnk, lhs_layout, rhs_layout)?;
                Ok(Storage::Cpu(storage))
            }
        }
    }

    pub fn affine(&self,
    layout: &Layout,
        mul: f64,
        add: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        match self {
            Self::Cpu(storage) => {
                let storage = storage.affine(layout, mul, add)?;
                Ok(Storage::Cpu(storage))
            }
        }
    }

    pub fn unary_op<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self, Box<dyn std::error::Error>> {
        match self {
            Self::Cpu(storage) => {
                let storage = storage.unary_op::<B>(layout)?;
                Ok(Storage::Cpu(storage))
            }
        }
    }
}
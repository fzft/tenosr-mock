extern crate libc;
use libc::{c_char, c_double, c_float, c_int};

mod ffi {
    use super::*;
    extern "C" {

        // C:=α⋅op(A)⋅op(B)+β⋅C
        // details: https://developer.apple.com/documentation/accelerate/1513282-cblas_dgemm
        #[link_name = "dgemm_"]
        pub fn dgemm_ffi(
            transa: *const c_char,
            transb: *const c_char,
            m: *const c_int,
            n: *const c_int,
            k: *const c_int,
            alpha: *const c_double,
            a: *const c_double,
            lda: *const c_int,
            b: *const c_double,
            ldb: *const c_int,
            beta: *const c_double,
            c: *mut c_double,
            ldc: *const c_int);

        pub fn srotg_(a: *mut c_float, b: *mut c_float, c: *mut c_float, s: *mut c_float);
    }

}

#[inline]
pub unsafe fn dgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &[f64],
    ldb: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
) {
    ffi::dgemm_ffi(
        &(transa as c_char),
        &(transb as c_char),
        &m,
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgemm() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];
        unsafe {
            dgemm(
                78,
                78,
                2,
                2,
                2,
                1.0,
                &a,
                2,
                &b,
                2,
                0.0,
                &mut c,
                2,
            );
        }
        assert_eq!(c, vec![7.0, 10.0, 15.0, 22.0]);
    }

    #[test]
    fn link() {
        unsafe {
            let mut a: f32 = 0.0;
            let mut b: f32 = 0.0;
            let mut c: f32 = 42.0;
            let mut d: f32 = 42.0;
            ffi::srotg_(
                &mut a as *mut _,
                &mut b as *mut _,
                &mut c as *mut _,
                &mut d as *mut _,
            );
            assert!(c == 1.0);
            assert!(d == 0.0);
        }
    }
}
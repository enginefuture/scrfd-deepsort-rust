use tch::Tensor;


#[derive(Clone)]
struct A<'a>{
    t:&'a Tensor
}

impl <'a>A<'a> {
    fn new(t:&'a Tensor) -> Self{
        A{
            t
        }
    }
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn test(){
        let t = Tensor::default();
        t.dot(&t);
        // let t = t.norm();
    }
}
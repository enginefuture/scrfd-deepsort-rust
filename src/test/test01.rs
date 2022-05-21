use tch::nn::{Module, OptimizerConfig};
use tch::{kind, nn, Device, Tensor};


fn my_module(p: nn::Path, dim:i64) -> impl nn::Module{
    let x1 = p.zeros("x1", &[dim]);
    let x2 = p.zeros("x2", &[dim]);
    nn::func(move |xs| xs * &x1 + xs.exp() * &x2)
}


fn gradient_descent(){
    let vs = nn::VarStore::new(Device::Cpu);
    let my_module = my_module(vs.root(), 7);
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    for _idx in 1..10{
        let xs = Tensor::zeros(&[7], kind::FLOAT_CPU);
        let ys = Tensor::zeros(&[7], kind::FLOAT_CPU);
        xs.print();
        ys.print();
        let loss = (my_module.forward(&xs) - ys).pow_tensor_scalar(2).sum(kind::Kind::Float);
        loss.print();
        opt.backward_step(&loss);
        loss.print();
        println!("==================");
    }
}


#[cfg(test)]
mod tests{

    use super::*;

    #[test]
    fn test(){
        let t = Tensor::of_slice(&[3.0, 4.0, 5.0, 7.0, 8.0]);
        let t1 = Tensor::of_slice(&[3.0, 4.0, 5.0, 7.0, 8.0]);

        let t = t * 2;
        let mut t1 = t1 * 2;
        let c = t.dot(&t1.t_());
        let d = f64::from(c.data());
        println!("{:?}", d);
        // let c = t.double_value(&[0]);
        // println!("{:?}", t.double_value(&[0]));
        // t.print()
    }


    #[test]
    fn test_nn(){
        gradient_descent()
    }
}
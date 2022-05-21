use std::ops::{Add, Div};

use tch::{
    nn::{
        self
        , Module
    }
    , Tensor
    , Device
};

#[derive(Debug)]
struct BasicBlock{
    is_downsample:bool,
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
    downsample: nn::SequentialT,
}


impl BasicBlock{

    pub fn new(vs: &nn::Path, c_in:i64, c_out:i64, is_downsample:bool) -> BasicBlock{
        let mut c1 = nn::ConvConfig::default();
        if is_downsample{
            c1.stride = 2;
            c1.padding = 1;
            c1.bias = false;
        }else{
            c1.stride = 1;
            c1.padding = 1;
            c1.bias = false;
        }
        let mut c2 = nn::ConvConfig::default();
        c2.stride = 1;
        c2.padding = 1;
        c2.bias = false;
        
        let mut basic = BasicBlock{
            is_downsample:is_downsample,
            conv1: nn::conv2d(vs, c_in, c_out, 3, c1),
            bn1: nn::batch_norm2d(vs, c_out, nn::BatchNormConfig::default()),
            conv2: nn::conv2d(vs, c_out, c_out, 3, c2),
            bn2: nn::batch_norm2d(vs, c_out, nn::BatchNormConfig::default()),
            downsample: nn::seq_t(),
        };

        let downsample = if is_downsample{
            let mut cd1 = nn::ConvConfig::default();
            cd1.stride = 2;
            cd1.bias = false;
            nn::seq_t()
                .add(nn::conv2d(vs, c_in, c_out, 1, cd1))
                .add(nn::batch_norm2d(vs, c_out, nn::BatchNormConfig::default()))
        }else if c_in != c_out{
            let mut cd1 = nn::ConvConfig::default();
            cd1.stride = 1;
            cd1.bias = false;
            basic.is_downsample = true;
            // println!("ttt:::{:?}", basic.is_downsample);
            nn::seq_t()
                .add(nn::conv2d(vs, c_in, c_out, 1, cd1))
                .add(nn::batch_norm2d(vs, c_out, nn::BatchNormConfig::default()))
        }else{
            nn::seq_t()
        };
        // println!("eee:::{:?}", basic.is_downsample);
        basic.downsample = downsample;
        
        basic
    }

}


impl nn::Module for BasicBlock {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let y = xs.apply(&self.conv1);
        let y = y.apply_t(&self.bn1, true);
        let y = y.relu();
        let y = y.apply(&self.conv2);
        let y = y.apply_t(&self.bn2, true);
        let mut x = Tensor::default();
        if self.is_downsample{
            x = xs.apply_t(&self.downsample, true);
            x = x.add(y);
            x = x.relu();
        }else{
            x = y.relu();
        }
        x
    }
}



fn make_layers(vs: &nn::Path, c_in:i64, c_out:i64, repeat_times: i64, is_downsample:bool) -> Vec<BasicBlock>{

    // let mut seq_t_v = nn::seq_t(); 
    let mut v = vec![];
    for i in 0..repeat_times{
        let basic = if i == 0{
            BasicBlock::new(vs, c_in, c_out, is_downsample)
        }else{
            BasicBlock::new(vs, c_out, c_out, false)
        };
        v.push(basic);
        // seq_t_v = seq_t_v.add(basic.seq_t());
    }
    // seq_t_v
    v
}

// Net网络
#[derive(Debug)]
pub struct Net{
    conv: nn::SequentialT,
    layer1: Vec<BasicBlock>,
    layer2: Vec<BasicBlock>,
    layer3: Vec<BasicBlock>,
    layer4: Vec<BasicBlock>,
    reid: bool,
    classifier: nn::SequentialT
}


impl Net{

    pub fn new(vs:nn::Path, num_classes: i64, reid: bool) -> Net{
        
        let mut net = Net::default();

        let mut c = nn::ConvConfig::default();
        c.stride = 1;
        c.padding = 1;
        net.conv = nn::seq_t()
                        .add(nn::conv2d(&vs, 3, 64, 3, c))
                        .add(nn::batch_norm2d(&vs, 64, nn::BatchNormConfig::default()))
                        .add_fn(|xs| xs.relu())
                        .add_fn(|xs| xs.max_pool2d(&[3], &[2], &[1], &[1], false))
                        ;
        
        net.layer1 = make_layers(&vs, 64, 64, 2, false);
        net.layer2 = make_layers(&vs, 64, 128, 2, true);
        net.layer3 = make_layers(&vs, 128, 256, 2, true);
        net.layer4 = make_layers(&vs, 256, 512, 2, true);
        
        net.reid = reid;
        
        net.classifier = nn::seq_t()
                                .add(nn::linear(&vs, 512, 256, nn::LinearConfig::default()))
                                .add(nn::batch_norm1d(&vs, 256, nn::BatchNormConfig::default()))
                                .add_fn(|xs| xs.relu())
                                .add_fn(|xs| xs.dropout(0.2, true))
                                .add(nn::linear(&vs, 256, num_classes, nn::LinearConfig::default()));

        net
    }
    
}

impl Default for Net {
    fn default() -> Self {
        Self { conv: nn::seq_t(), layer1: Default::default(), layer2: Default::default(), layer3: Default::default(), layer4: Default::default(), reid: Default::default(), classifier: nn::seq_t() }
    }
}

impl nn::Module for Net{
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut x = xs.apply_t(&self.conv, true);
        for item in &self.layer1{
            x = item.forward(&x);
        }
        for item in &self.layer2{
            x = item.forward(&x);
        }
        for item in &self.layer3{
            x = item.forward(&x);
        }
        for item in &self.layer4{
            x = item.forward(&x);
        }
        // let x = x.apply_t(&self.layer1, true);
        // let x = x.apply_t(&self.layer2, true);
        // let x = x.apply_t(&self.layer3, true);
        // let x = x.apply_t(&self.layer4, true);
        let x = x.avg_pool2d(&[8, 4], &[1], &[0], false, true, None);
        let x = x.view((x.size()[0], -1));
        if self.reid{
            let xn = x.norm_out(&x, 2, &[1], true);
            let x = x.div(xn);
            return x;
        }
        let x = x.apply_t(&self.classifier, true);
        return x;
    }
}



#[cfg(test)]
mod tests{
    use tch::{};

    use super::*;

    // 测试网络
    #[test]
    fn test_forward(){
        let device = Device::cuda_if_available();
        let mut vs = nn::VarStore::new(device);

        let net = Net::new(vs.root(), 512, false);
        let x = Tensor::randn(&[1, 3, 128, 64], (tch::Kind::Float, device));
        let y = net.forward(&x);
        println!("{:?}", y);
        println!("{:?}", y.print());
        
    }

    #[test]
    fn test(){
        let device = Device::cuda_if_available();
        let mut vs = nn::VarStore::new(device);

        let net = Net::new(vs.root(), 751, false);
        let x = Tensor::randn(&[4, 3, 128, 64], (tch::Kind::Float, device));
        println!("{:?}", x);
        let mut y = x.apply_t(&net.conv, true);
        
        for item in net.layer1{
            y = item.forward(&y);
        }
        let t1 = &y;
        // println!("{:?}", t1);
        let mut y1 = Tensor::default();
        for item in net.layer2{
            y = item.forward(&y);
            // y1 = y.apply(&item.conv1);
            // y1 = y1.apply_t(&item.bn1, true);
            // y1 = y1.relu();
            // y1 = y1.apply(&item.conv2);
            // y1 = y1.apply_t(&item.bn2, true);
            // let mut x1 = Tensor::default();
            // println!("{:#?}", item.is_downsample);
            // if item.is_downsample{
            //     x1 = t1.apply_t(&item.downsample, true);
            // };
            // x1 = x1.add(y1);
            // y = x1.relu();
            
            // break;
        }

        for item in net.layer3{
            y = item.forward(&y);
        }
        for item in net.layer4{
            y = item.forward(&y);
        }
        println!("{:#?}", y);
        // let y = y.apply_t(&net.layer2, true);
        // let y = net.forward(&x);
        // let t = net.seq_t();
        // let y = x.apply_t(&t, true);
        // println!("{:#?}", y1);
    }
}
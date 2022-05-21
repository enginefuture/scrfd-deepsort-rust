
fn vec_dot(v1: &Vec<Vec<f64>>, v2: &Vec<Vec<f64>>) -> Vec<Vec<f64>>{
    let mut result = vec![];
    for i in 0..v1.len(){
        let mut sub_res = vec![];
        for j in 0..v1[0].len(){
            let v = v1[i][0] * v2[0][i] 
                    + v1[i][1] * v2[1][i]
                    + v1[i][2] * v2[2][i]
                    + v1[i][3] * v2[3][i]
                    + v1[i][4] * v2[4][i]
                    + v1[i][5] * v2[5][i]
                    + v1[i][6] * v2[6][i]
                    + v1[i][7] * v2[7][i]
                    + v1[i][8] * v2[8][i]
                    + v1[i][9] * v2[9][i]
                    + v1[i][10] * v2[10][i]
                    + v1[i][11] * v2[11][i]
                    + v1[i][12] * v2[12][i]
                    + v1[i][13] * v2[13][i]
                    + v1[i][14] * v2[14][i]
                    + v1[i][15] * v2[15][i]
                    + v1[i][16] * v2[16][i]
                    + v1[i][17] * v2[17][i]
                    + v1[i][18] * v2[18][i]
                    + v1[i][19] * v2[19][i]
                    + v1[i][20] * v2[20][i]
                    + v1[i][11] * v2[21][i]
                    + v1[i][22] * v2[22][i]
                    + v1[i][23] * v2[23][i]
                    + v1[i][24] * v2[24][i]
                    + v1[i][25] * v2[25][i]
                    + v1[i][26] * v2[26][i]
                    + v1[i][27] * v2[27][i]
                    + v1[i][28] * v2[28][i]
                    + v1[i][29] * v2[29][i]
                    + v1[i][30] * v2[30][i]
                    + v1[i][31] * v2[31][i]
                    + v1[i][32] * v2[32][i]
                    + v1[i][33] * v2[33][i]
                    + v1[i][34] * v2[34][i]
                    + v1[i][35] * v2[35][i]
                    + v1[i][36] * v2[36][i]
                    + v1[i][37] * v2[37][i]
                    + v1[i][38] * v2[38][i]
                    + v1[i][39] * v2[39][i]
                    + v1[i][40] * v2[40][i]
                    + v1[i][41] * v2[41][i]
                    + v1[i][42] * v2[42][i]
                    + v1[i][43] * v2[43][i]
                    + v1[i][44] * v2[44][i]
                    + v1[i][45] * v2[45][i]
                    + v1[i][46] * v2[46][i]
                    + v1[i][47] * v2[47][i]
                    + v1[i][48] * v2[48][i]
                    + v1[i][49] * v2[49][i]
            ;
            sub_res.push(v);
        }
        result.push(sub_res);
    }
    result
}

#[cfg(test)]
mod tests{
    use chrono::{DateTime, Local};
    use rand::Rng;
    use ndarray as np;
    use crate::test::test03::vec_dot;

    #[test]
    fn test(){
        let mut v1 = vec![];
        let mut v2 = vec![];
        let mut a = np::Array::<f64, _>::zeros((50, 50));
        let mut b = np::Array::<f64, _>::zeros((50, 50));


        for i in 0..50{
            let mut sub_v1 = vec![];
            let mut sub_v2 = vec![];
            for j in 0..50{
                let mut rng = rand::thread_rng();
                let v = rng.gen_range(-10000.0..1000000.0);
                sub_v1.push(v);
                let va = a.get_mut((i, j)).unwrap();
                *va = v;
                let mut rng = rand::thread_rng();
                let v = rng.gen_range(-100000.0..100000.0);
                sub_v2.push(v);
                let vb = b.get_mut((i, j)).unwrap();
                *vb = v;
            }
            
            v1.push(sub_v1);
            v2.push(sub_v2);
        }
        let mut sum = 0;
        let mut result = vec![];
        for _ in 0..10{
            let start: DateTime<Local> = Local::now();
            let m1: i64 = start.timestamp_millis();
            result = vec_dot(&v1, &v2);
            let end: DateTime<Local> = Local::now();
            let m2: i64 = end.timestamp_millis();
    
            println!("{:?}-{:?}={:?}", m2, m1, m2 - m1);
            sum += m2 - m1;
        }
        
        let mut sum_nd = 0;
        let mut resutl_nd = np::Array::<f64, _>::zeros((50, 50));

        for _ in 0..10{
            let start: DateTime<Local> = Local::now();
            let m1: i64 = start.timestamp_millis();
            resutl_nd = a.dot(&b);
            let end: DateTime<Local> = Local::now();
            let m2: i64 = end.timestamp_millis();
    
            println!("{:?}-{:?}={:?}", m2, m1, m2 - m1);
            sum_nd += m2 - m1;
        }
        println!("{:?}", a.get((0,0)));
        println!("{:?}", b.get((0,0)));
        println!("{:?}", v1[0][0]);
        println!("{:?}", v2[0][0]);
        println!("{:?}", result[0][0]);
        println!("{:?}", resutl_nd.get((0,0)));
        println!("sum={:?};::::sum_nd={:?}", sum, sum_nd);

    }
}
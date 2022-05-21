

pub fn non_max_suppression(boxes:Vec<[f64;4]>, max_bbox_overlap:f64, scores:Vec<f64>) ->Vec<usize>{
    let mut pick = vec![];
    if boxes.len() == 0{
        return pick;
    }
    let mut x1 = vec![];
    let mut y1 = vec![];
    let mut x2 = vec![];
    let mut y2 = vec![];
    let mut area = vec![];
    for i in boxes{
        x1.push(i[0]);
        y1.push(i[1]);
        x2.push(i[2] + i[0]);
        y2.push(i[3] + i[1]);
        let a = (i[2] + 1.0) * (i[3] + 1.0);
        area.push(a);
    }
    // scores 最小元素下标
    let mut idxs:Vec<usize> = vec![];

    if scores.len() != 0{
        let l = scores.len();
        let d = (0..l).collect::<Vec<usize>>();
        idxs = (0..l).collect::<Vec<usize>>();
        idxs.sort_by(|x1, x2| x2.partial_cmp(x1).unwrap())
    }else {
        let l = y2.len();
        idxs = (0..l).collect::<Vec<usize>>();
        idxs.sort_by(|x1, x2| x2.partial_cmp(x1).unwrap())
    }

    while idxs.len() > 0 {
        let last = idxs.len() - 1;
        let i = idxs[last];
        pick.push(i);
        
        let mut xx1 = vec![];
        let mut yy1 = vec![];
        let mut xx2 = vec![];
        let mut yy2 = vec![];
        let mut area_re = vec![];
        for j in &mut idxs[0..last]{
            if x1[i] > x1[*j]{
                xx1.push(x1[i]);
            }else {
                xx1.push(x1[*j]);
            }
            if y1[i] > y1[*j]{
                yy1.push(x1[i]);
            }else {
                yy1.push(x1[*j]);
            }
            if x2[i] > x2[*j]{
                xx2.push(x1[i]);
            }else {
                xx2.push(x1[*j]);
            }
            if y2[i] > y2[*j]{
                yy2.push(x1[i]);
            }else {
                yy2.push(x1[*j]);
            }

            area_re.push(area[*j]);
        }
        let mut w = vec![];
        let mut h = vec![];
        for j in 0..xx1.len(){
            if 0.0 > xx2[j] - xx1[j] + 1.0{
                w.push(0.0)
            }else {
                w.push(xx2[j] - xx1[j] + 1.0)
            }

            if 0.0 > yy2[j] - yy1[j] + 1.0{
                h.push(0.0)
            }else {
                h.push(yy2[j] - yy1[j] + 1.0)
            }
        }

        let mut overlap = vec![];
        for j in 0..w.len(){
            overlap.push((w[j] * h[j]) / area_re[j]);
        }
        // 
        idxs.remove(last);
        for j in 0..overlap.len(){
            if overlap[j] > max_bbox_overlap{
                idxs.remove(j);
            }
        }    
    }
    pick
}



#[cfg(test)]
mod tests{

    #[test]
    fn test(){
        let mut l1 = vec![];
        l1.push(1);
        let l = l1.len() as i32;
        let d = (0..l).collect::<Vec<i32>>();
        let mut t = (0..10).collect::<Vec<i32>>().sort_by(|x1, x2| x2.partial_cmp(x1).unwrap());
        // t.sort_by(|x1, x2| x2.partial_cmp(x1).unwrap());
        println!("{:?}", d);
    }
}
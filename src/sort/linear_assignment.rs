use super::{
    track::{Track, self}
    , detection::Detection
    , tracker::Tracker
    , kalman_filter::{
        KalmanFilter
        , CHI2INV95
    }, iou_matching::iou_cost
};

pub const INFTY_COST: f64 = 1e+5;

pub fn distance_metric(
    tracker: &mut Tracker
    , tracks: Vec<Track>
    , dets: Vec<Detection>
    , mut track_indices: Vec<usize>
    , mut detection_indices: Vec<usize>
) -> Vec<Vec<f64>>{

    let mut features = vec![];
    for i in detection_indices.clone(){
        features.push(dets[i].feature.clone());
    }

    let mut targets = vec![];
    for i in track_indices.clone(){
        targets.push(tracks[i].track_id)
    }

    let cost_matrix = tracker.metric.distance(features, targets.clone());

    let cost_matrix = gate_cost_matrix(tracker.kf.clone(), cost_matrix, tracks, dets, track_indices.clone(), detection_indices.clone(), INFTY_COST, false);

    cost_matrix
}

pub fn linear_sum_assignment(cost_matrix: Vec<Vec<f64>>, maximize: bool) -> (Vec<usize>, Vec<usize>){


    if maximize{
        // 变负
        // cost_matrix = - cost_matrix
    }
    // 取出最小的n*n矩阵

    let (row, col) = (cost_matrix.len(), cost_matrix[0].len());

    let mut a = vec![];
    let mut b = vec![];

    if col < row{
        for i in 0..col{
            let mut ind = 0;
            let mut min = cost_matrix[0][i];
            let mut cont = false;
            for j in 0..row{
                cont = false;
                if min >= cost_matrix[j][i]{
                    for v in b.clone(){
                        if v == j{
                            cont = true;
                            break;
                        }
                    }
                    if cont{
                        continue;
                    }
                    ind = j;
                    min = cost_matrix[j][i];
    
                }
            }
            a.push(ind);
            b.push(i);
    
        }
    }else{
        for i in 0..row{
            let mut ind = 0;
            let mut min = cost_matrix[i][0];
            let mut cont = false;
            for j in 0..col{
                cont = false;
                if min >= cost_matrix[i][j]{
                    for v in b.clone(){
                        if v == j{
                            cont = true;
                            break;
                        }
                    }
                    if cont{
                        continue;
                    }
                    ind = j;
                    min = cost_matrix[i][j];
    
                }
            }
            a.push(i);
            b.push(ind);
    
        }
    }
    (a, b)
}

pub fn min_cost_matching(
    tracker: &mut Tracker
    , t:String
    , max_distance: f64
    , tracks: Vec<Track>
    , detections: Vec<Detection>
    , mut track_indices: Vec<usize>
    , mut detection_indices: Vec<usize>
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>){

    let mut matches = vec![];
    // None
    if track_indices.len() < 0{
        for i in 0..tracks.len(){
            track_indices.push(i);
        }
    }

    if detection_indices.len() < 0{
        for i in 0..detections.len(){
            detection_indices.push(i);
        }
    }
    // 返回值
    if detection_indices.len() == 0 || track_indices.len() == 0{
        return (matches, track_indices, detection_indices);
    }

    let mut cost_matrix = vec![];
    if t.eq("distance_metric"){
        cost_matrix = distance_metric(tracker, tracks, detections, track_indices.clone(), detection_indices.clone());
    }else if t.eq("iou_cost") {
        cost_matrix = iou_cost(tracks, detections, track_indices.clone(), detection_indices.clone());
    }

    for i in 0..cost_matrix.len(){
        for j in 0..cost_matrix[i].len(){
            if cost_matrix[i][j] > max_distance{
                cost_matrix[i][j] = max_distance + 1e-5;
            }
        }
    }

    // （*****************************************************）
    // cost_matrix.dedup();
    // （*****************************************************）

    let (row_indices, col_indices) = linear_sum_assignment(cost_matrix.clone(), false);

    let mut unmatched_tracks = vec![];
    let mut unmatched_detections = vec![];

    for col in 0..detection_indices.len(){
        let mut b = false;
        
        for c in col_indices.clone(){
            if c == col{
                b = true;
                break;
            }
        }
        if !b{
            unmatched_detections.push(detection_indices[col]);
        }

    }
    for row in 0..track_indices.len(){
        let mut b = false;
        for r in row_indices.clone(){
            if r == row{
                b =true;
            }
        }
        if !b{
            unmatched_tracks.push(track_indices[row]);
        }
    }

    for i in 0..row_indices.len(){
        let row = row_indices[i];
        let col = col_indices[i];
        let track_idx = track_indices[row];
        let detection_idx = detection_indices[col];

        if cost_matrix[row][col] > max_distance{
            unmatched_tracks.push(track_idx);
            unmatched_detections.push(detection_idx);
        }else {
            
            matches.push((track_idx, detection_idx));
        }
    }

    // matches.dedup_by(|a, b| a.1==b.1);
    // 临时代码 
    // unmatched_detections.dedup();
    // matches.dedup();
    // let udcount = if detection_indices.len() > unmatched_detections.len(){unmatched_detections.len()}else{detection_indices.len()};
    // let mdcount = if detection_indices.len() > matches.len(){matches.len()}else{detection_indices.len()};
    // let mut ud = vec![];
    // for i in 0..udcount{
    //     ud.push(unmatched_detections[i]);
    // }
    // let mut mt = vec![];
    // for i in 0..mdcount{
    //     mt.push(matches[i]);
    // }

    // let matches = mt;
    // let unmatched_detections = ud;
    // 临时代码

    (matches, unmatched_tracks, unmatched_detections)
    
}


pub fn matching_cascade(
    tracker: &mut Tracker
    , max_distance: f64
    , cascade_depth: i32
    , tracks: Vec<Track>
    , detections: Vec<Detection>
    , track_indices: Vec<usize>
    , detection_indices: Vec<usize>
)-> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>){
    // let mut track_indices_mut = vec![];
    // // python判断None 传值都不为None
    // if track_indices.len() < 0{
    //     for i in 0..tracks.len(){
    //         track_indices_mut.push(i);
    //     }
    // }
    // let track_indices = track_indices_mut;

    let mut detection_indices_mut = vec![];
    if detection_indices.len() == 0{
        for i in 0..detections.len(){
            detection_indices_mut.push(i);
        }
    }
    let detection_indices = detection_indices_mut;

    let mut unmatched_detections = detection_indices.clone();
    let mut matches = vec![];
    for level in 0..cascade_depth{
        if unmatched_detections.len()  == 0{
            break;
        }

        let mut track_indices_l = vec![];
        for k in track_indices.clone(){
            if tracks[k].time_since_update == 1 + level{
                track_indices_l.push(k);
            }
        }

        if track_indices_l.len() == 0{
            continue;
        }
        
        let (matches_l, _, unmatched_detections_l) = 
                min_cost_matching(tracker, "distance_metric".to_string(),max_distance, tracks.clone(), detections.clone(), track_indices_l.clone(), unmatched_detections.clone());
        
        unmatched_detections = unmatched_detections_l;
        for i in matches_l{
            matches.push(i);
        }
    }

    // track_indices
    let mut unmatched_tracks = vec![];
    for i in track_indices{
        unmatched_tracks.push(i);
    }
    unmatched_tracks.dedup();

    for (k, _) in matches.clone(){
        // unmatched_tracks.push(k);
        let mut ind_v = vec![];
        for i in 0..unmatched_tracks.len(){
            if unmatched_tracks[i] == k{
                ind_v.push(i);
            }
        }
        ind_v.reverse();
        for i in ind_v{
            unmatched_tracks.remove(i);
        }

    }

    // unmatched_tracks.dedup();

    (matches, unmatched_tracks, unmatched_detections)

}


pub fn gate_cost_matrix(
    kf: KalmanFilter
    , mut cost_matrix: Vec<Vec<f64>>
    , tracks: Vec<Track>
    , dets: Vec<Detection>
    , track_indices: Vec<usize>
    , detection_indices: Vec<usize>
    , gated_cost: f64
    , only_position: bool
) -> Vec<Vec<f64>>{
    let gating_dim = if only_position{1}else{3};
    let gating_threshold = CHI2INV95[gating_dim];
    

    let mut measurements = vec![];
    for i in detection_indices{
        let v = dets[i].to_xyah();
        measurements.push(v.to_vec());
    }
    for i in 0..track_indices.len(){
        let track = tracks[track_indices[i]].clone();

        let gating_distance = kf.gating_distance(&track.mean, &track.covariance, measurements.clone(), only_position);

        for j in 0..cost_matrix[i].len(){
            if gating_distance[j] > gating_threshold{
                cost_matrix[i][j] = gated_cost;
            }
            
        }
    }
    cost_matrix
}


#[cfg(test)]
mod tests{
    
    #[test]
    fn test(){
        println!("nihoa");
        let mut i = vec![];
        i.push(1);
        i.push(1);
        i.push(1);
        i.push(2);
        i.push(2);
        i.push(2);
        i.dedup();
        println!("{:?}", i);
    }
}
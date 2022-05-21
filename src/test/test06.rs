

#[cfg(test)]
mod tests{

    #[test]
    fn test(){
        let mut m = vec![];
        m.push((1, 0));
        m.push((2, 0));
        m.push((3, 0));
        m.push((4, 0));
        m.push((5, 0));
        m.dedup_by(|a, b| a.1 ==b.1);
        println!("{:?}", m);
    }

}
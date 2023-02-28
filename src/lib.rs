pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[no_mangle]
pub extern "C" fn add_five(input: i32) -> i32 {
    input + 5
}

#[no_mangle]
pub extern "C" fn concat_string(input: &str) -> String {
    // concat!("This is the input: ", input);
    let mut data = String::from("this is input : ");
    let input_char: Vec<char> = input.chars().collect();
    for c in input_char {
        data.push(c)
    }
    data.to_string()
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn add_five_test() {
        let result = add_five(15);
        assert_eq!(result, 20);
    }

    #[test]
    fn concat_string_test() {
        let result = concat_string("input");
        assert_eq!(result, String::from("this is input : input"));
    }
}

use wasm_bindgen::prelude::*;

#[no_mangle]
pub extern "C" fn add_five(input: i32) -> i32 {
    input + 5
}

#[wasm_bindgen]
pub extern "C" fn concat_string(input: String) -> String {
    // concat!("This is the input: ", input);
    let data = format!("this is input : {}", input);
    return data.into();
}

#[wasm_bindgen]
pub extern "C" fn object_detection(row: i32, col:i32, data: Vec<i8>) -> i32 {
    let mut total:i32 = 0;
    print!("({}, {}) = {}", row, col, data.len());
    for val in data {
        total += i32::from(val);
    }
    return total;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_five_test() {
        let result = add_five(15);
        assert_eq!(result, 20);
    }

    #[test]
    fn concat_string_test() {
        let result = concat_string(String::from("input"));
        assert_eq!(result, String::from("this is input : input"));
    }

    #[test]
    fn object_detection_test() {
        let mut input: Vec<i8> = Vec::new();

        for i in [1, 2, 3, 4, 5] {
            for j in [1, 2, 3, 4, 5] {
                input.push(i8::from(i*j))
            }
        }
        let result = object_detection(5, 5, input);
        assert_eq!(result, 225);
    }
}

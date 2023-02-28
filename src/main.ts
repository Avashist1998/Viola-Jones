

const data = { selectedImageID: ""};


(async () => {
let response = await fetch("my-crate/pkg/my_crate_bg.wasm")
let bytes = await response.arrayBuffer();
let { instance } = await WebAssembly.instantiate(bytes, { });
const add_five = instance.exports.add_five as CallableFunction;
const concat_string = instance.exports.concat_string as CallableFunction;

console.log("The answer is: ", add_five(13));
console.log("The answer is : ", add_five(25));
console.log(concat_string("This is a test"));

const loading = document.getElementById("loading")
const content = document.getElementById("content")
if (loading !== null) {
    loading.className = "loadingHide"
}
if (content !== null) {
    content.className = "contentShowing"
}
})();

function imageSelect(id: string) {
    const selectedImg = document.getElementById(id)
    if (selectedImg !== null) {
        selectedImg.className = "selectedImg"
    }
    if (data.selectedImageID !== "") {
        const unselectedImg = document.getElementById(data.selectedImageID)
        if (unselectedImg !== null) {
            unselectedImg.className = "unselectedImg"
        }
    }
    const imageSubmitBtn = document.getElementById("imageSubmitBtn") as HTMLButtonElement
    if (imageSubmitBtn !== null) {
        imageSubmitBtn.disabled = false
    }
    data.selectedImageID = id
}

function iCanSubmit() {
    console.log(`Submitting Image ${data.selectedImageID}`)
    // console.log(instance.exports.concat_string(selectedImageID));
}

var colorImage = document.getElementById("color") as HTMLImageElement;
colorImage.addEventListener("click", () => {imageSelect("color")});
var groupImage = document.getElementById("group") as HTMLImageElement;
groupImage.addEventListener("click", () => {imageSelect("group")});
var blackAndWhiteImage = document.getElementById("black&white") as HTMLImageElement;
blackAndWhiteImage.addEventListener("click", () => {imageSelect("black&white")});

const submitButton = document.getElementById("imageSubmitBtn") as HTMLButtonElement;
submitButton.addEventListener("click", iCanSubmit);
submitButton.disabled = true;

export { imageSelect, iCanSubmit}
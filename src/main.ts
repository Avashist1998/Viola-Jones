import wasmInit, {
    concat_string
} from "../my-crate/pkg/my_crate.js"

const data = { selectedImageID: ""};

const runWasm = async () => {
    // Instantiate our wasm module
    const rustWasm = await wasmInit("my-crate/pkg/my_crate_bg.wasm");  
};

runWasm().then( () => {
    const loading = document.getElementById("loading")
    const content = document.getElementById("content")
    if (loading !== null) {
        loading.className = "loadingHide"
    }
    if (content !== null) {
        content.className = "contentShowing"
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
});


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
    console.log(concat_string(data.selectedImageID))
}

export { imageSelect, iCanSubmit}
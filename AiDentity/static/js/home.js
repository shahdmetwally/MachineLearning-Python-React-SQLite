function previewImage(input) {
  var previewContainer = document.getElementById("image-preview-container");
  var preview = document.getElementById("image-preview");
  var waitingMessage = document.getElementById("waiting-message");
  var file = input.files[0];
  var reader = new FileReader();

  reader.onload = function (e) {
    preview.src = e.target.result;
    preview.style.display = "block";
    waitingMessage.style.display = "none";
  };

  if (file) {
    reader.readAsDataURL(file);
  } else {
    // reset preview if no file is selected
    preview.src = "#";
    preview.style.display = "none";
    waitingMessage.style.display = "block";
  }
}

// add event listener to change function
document.getElementById("id_image").addEventListener("change", function () {
  previewImage(this);
});

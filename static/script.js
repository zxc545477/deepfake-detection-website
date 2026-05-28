document.addEventListener("DOMContentLoaded", function () {
  const watchBtn = document.getElementById("watchDemoBtn");
  const modal = document.getElementById("demoModal");
  const closeBtn = document.getElementById("closeModal");

  watchBtn.addEventListener("click", function () {
    modal.classList.remove("hidden");
  });

  closeBtn.addEventListener("click", function () {
    modal.classList.add("hidden");
  });

  modal.addEventListener("click", function (e) {
    if (e.target === modal) {
      modal.classList.add("hidden");
    }
  });
});
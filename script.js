(() => {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Scroll-reveal
  const items = document.querySelectorAll(".reveal");
  if (reduce || !("IntersectionObserver" in window)) {
    items.forEach(el => el.classList.add("in"));
  } else {
    document.documentElement.classList.add("js");
    const io = new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
    }), { threshold: 0.12 });
    items.forEach(el => io.observe(el));
  }

  // Mobile menu
  const nav = document.getElementById("nav");
  const toggle = nav.querySelector(".nav-toggle");
  const icon = toggle.querySelector(".bi");
  const setOpen = open => {
    nav.classList.toggle("open", open);
    toggle.setAttribute("aria-expanded", open);
    icon.className = "bi " + (open ? "bi-x-lg" : "bi-list");
  };
  toggle.addEventListener("click", () => setOpen(!nav.classList.contains("open")));
  nav.querySelectorAll(".nav-links a").forEach(a => a.addEventListener("click", () => setOpen(false)));
  document.addEventListener("click", e => { if (!nav.contains(e.target)) setOpen(false); });
  document.addEventListener("keydown", e => { if (e.key === "Escape") setOpen(false); });

  // Active nav link
  const links = [...document.querySelectorAll("nav .nav-links a")];
  if ("IntersectionObserver" in window) {
    const so = new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting) links.forEach(a =>
        a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
    }), { rootMargin: "-40% 0px -55% 0px" });
    links.forEach(a => { const s = document.querySelector(a.getAttribute("href")); if (s) so.observe(s); });
  }
})();

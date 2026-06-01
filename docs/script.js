const storageKey = "site-theme";
const body = document.body;
const themeToggle = document.querySelector(".theme-toggle");
const themeToggleText = document.querySelector(".theme-toggle-text");
const menuToggle = document.querySelector(".menu-toggle");
const siteNav = document.querySelector(".site-nav");
const header = document.querySelector(".site-header");
const yearElement = document.getElementById("year");
const navLinks = Array.from(document.querySelectorAll(".nav-link"));
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

function setTheme(theme) {
  if (theme === "light") {
    body.setAttribute("data-theme", "light");
    themeToggleText.textContent = "Light";
  } else {
    body.removeAttribute("data-theme");
    themeToggleText.textContent = "Dark";
  }
  localStorage.setItem(storageKey, theme);
}

function getPreferredTheme() {
  const storedTheme = localStorage.getItem(storageKey);
  if (storedTheme) {
    return storedTheme;
  }
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function toggleMenu(forceClose = false) {
  const shouldOpen = forceClose ? false : !siteNav.classList.contains("is-open");
  siteNav.classList.toggle("is-open", shouldOpen);
  menuToggle.setAttribute("aria-expanded", String(shouldOpen));
}

function updateHeaderState() {
  header.classList.toggle("is-scrolled", window.scrollY > 16);
}

function setActiveLink(id) {
  navLinks.forEach((link) => {
    const isActive = link.getAttribute("href") === `#${id}`;
    link.classList.toggle("is-active", isActive);
  });
}

if (sections.length > 0) {
  const observer = new IntersectionObserver(
    (entries) => {
      const visibleEntry = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (visibleEntry) {
        setActiveLink(visibleEntry.target.id);
      }
    },
    {
      rootMargin: "-35% 0px -45% 0px",
      threshold: [0.2, 0.4, 0.65]
    }
  );

  sections.forEach((section) => observer.observe(section));
}

themeToggle?.addEventListener("click", () => {
  const nextTheme = body.getAttribute("data-theme") === "light" ? "dark" : "light";
  setTheme(nextTheme);
});

menuToggle?.addEventListener("click", () => {
  toggleMenu();
});

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    toggleMenu(true);
  });
});

window.addEventListener("scroll", updateHeaderState, { passive: true });
window.addEventListener("resize", () => {
  if (window.innerWidth > 960) {
    toggleMenu(true);
  }
});

document.addEventListener("click", (event) => {
  if (!siteNav.contains(event.target) && !menuToggle.contains(event.target) && siteNav.classList.contains("is-open")) {
    toggleMenu(true);
  }
});

setTheme(getPreferredTheme());
updateHeaderState();
setActiveLink("about");

if (yearElement) {
  yearElement.textContent = new Date().getFullYear();
}

const storageKey = "site-theme";
const body = document.body;
const themeToggle = document.querySelector(".theme-toggle");
const themeToggleText = document.querySelector(".theme-toggle-text");
const tocToggle = document.querySelector(".toc-toggle");
const noteSidebar = document.getElementById("note-sidebar");
const noteTitle = document.getElementById("note-title");
const noteHeading = document.getElementById("note-heading");
const notePath = document.getElementById("note-path");
const noteContent = document.getElementById("note-content");
const tocNav = document.getElementById("toc-nav");
const header = document.querySelector(".site-header");
const embeddedMarkdown = document.getElementById("embedded-markdown");

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

function updateHeaderState() {
  header.classList.toggle("is-scrolled", window.scrollY > 16);
}

function toggleSidebar(forceClose = false) {
  const shouldOpen = forceClose ? false : !noteSidebar.classList.contains("is-open");
  noteSidebar.classList.toggle("is-open", shouldOpen);
  tocToggle.setAttribute("aria-expanded", String(shouldOpen));
}

function sanitizeNotePath(rawPath) {
  const path = rawPath || "rl/ppo.md";
  if (
    path.includes("..") ||
    path.startsWith("/") ||
    path.startsWith("http://") ||
    path.startsWith("https://") ||
    !path.endsWith(".md")
  ) {
    return "rl/ppo.md";
  }
  return path;
}

function slugify(text, usedSlugs) {
  const base = text
    .toLowerCase()
    .trim()
    .replace(/[^\w\u4e00-\u9fff\s-]/g, "")
    .replace(/\s+/g, "-");

  let slug = base || "section";
  let counter = 2;

  while (usedSlugs.has(slug)) {
    slug = `${base || "section"}-${counter}`;
    counter += 1;
  }

  usedSlugs.add(slug);
  return slug;
}

function preprocessMarkdown(markdown) {
  const blockMathPattern = /\\\[\s*([\s\S]*?)\s*\\\]/g;

  function isSimpleInlineMath(content) {
    const normalized = content
      .replace(/\s+/g, " ")
      .trim();

    if (normalized.length === 0 || normalized.length > 24) {
      return false;
    }

    return !/[=+\-/*<>]/.test(normalized) &&
      !/\\frac|\\dfrac|\\cfrac|\\sum|\\prod|\\int|\\min|\\max|\\mathbb|\\left|\\right/.test(normalized);
  }

  return markdown
    .replace(blockMathPattern, (_, content) => {
      const normalized = content.replace(/\s+/g, " ").trim();
      if (isSimpleInlineMath(normalized)) {
        return `<span class="inline-math-strong">\\(${normalized}\\)</span>`;
      }
      return `$$\n${content.trim()}\n$$`;
    })
    .replace(/\\\(/g, "$")
    .replace(/\\\)/g, "$");
}

function buildToc() {
  const headings = Array.from(noteContent.querySelectorAll("h1, h2, h3, h4, h5, h6"));

  if (headings.length === 0) {
    tocNav.innerHTML = '<p class="toc-empty">No headings found in this note.</p>';
    return;
  }

  tocNav.innerHTML = "";

  headings.forEach((heading) => {
    const link = document.createElement("a");
    const level = Number(heading.tagName.replace("H", ""));
    link.className = `toc-link level-${level}`;
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent;
    link.addEventListener("click", () => {
      if (window.innerWidth <= 960) {
        toggleSidebar(true);
      }
    });
    tocNav.appendChild(link);
  });

  const tocLinks = Array.from(tocNav.querySelectorAll(".toc-link"));
  const observer = new IntersectionObserver(
    (entries) => {
      const activeEntry = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (!activeEntry) {
        return;
      }

      tocLinks.forEach((link) => {
        link.classList.toggle("is-active", link.getAttribute("href") === `#${activeEntry.target.id}`);
      });
    },
    {
      rootMargin: "-20% 0px -65% 0px",
      threshold: [0.1, 0.35, 0.6]
    }
  );

  headings.forEach((heading) => observer.observe(heading));
}

function renderMath() {
  if (typeof renderMathInElement !== "function") {
    return;
  }

  renderMathInElement(noteContent, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false }
    ],
    throwOnError: false
  });
}

function highlightCodeBlocks() {
  if (typeof hljs === "undefined") {
    return;
  }

  noteContent.querySelectorAll("pre code").forEach((block) => {
    hljs.highlightElement(block);
  });
}

function renderMarkdown(markdown, path) {
  const normalizedMarkdown = markdown.trim().length > 0
    ? preprocessMarkdown(markdown)
    : "# Empty note\n\nThis markdown file exists, but it does not contain content yet.";

  if (typeof marked?.parse !== "function") {
    noteContent.innerHTML = "";
    const fallback = document.createElement("pre");
    fallback.textContent = normalizedMarkdown;
    noteContent.appendChild(fallback);
    noteTitle.textContent = path.split("/").pop();
    noteHeading.textContent = "Markdown preview unavailable";
    document.title = `${noteTitle.textContent} | Note Viewer`;
    return;
  }

  noteContent.innerHTML = marked.parse(normalizedMarkdown, {
    gfm: true,
    breaks: false
  });

  const usedSlugs = new Set();
  const headings = Array.from(noteContent.querySelectorAll("h1, h2, h3, h4, h5, h6"));
  headings.forEach((heading) => {
    heading.id = slugify(heading.textContent, usedSlugs);
  });

  const firstHeading = headings[0]?.textContent || path.split("/").pop().replace(".md", "");
  noteTitle.textContent = firstHeading;
  noteHeading.textContent = firstHeading;
  document.title = `${firstHeading} | Note Viewer`;

  highlightCodeBlocks();
  renderMath();
  buildToc();
}

async function loadNote() {
  const params = new URLSearchParams(window.location.search);
  const path = sanitizeNotePath(body.dataset.notePath || params.get("path"));
  notePath.textContent = path;

  if (embeddedMarkdown) {
    renderMarkdown(embeddedMarkdown.textContent, path);
    return;
  }

  try {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load ${path}`);
    }

    const rawMarkdown = await response.text();
    renderMarkdown(rawMarkdown, path);
  } catch (error) {
    noteTitle.textContent = "Unable to load note";
    noteHeading.textContent = "Unable to load note";
    const detail = window.location.protocol === "file:"
      ? `${error.message}. Local file previews cannot fetch Markdown. Open a note page with embedded content or run a local web server.`
      : error.message;
    noteContent.innerHTML = `<p class="note-error">${detail}</p>`;
    tocNav.innerHTML = '<p class="toc-empty">No table of contents available.</p>';
  }
}

themeToggle?.addEventListener("click", () => {
  const nextTheme = body.getAttribute("data-theme") === "light" ? "dark" : "light";
  setTheme(nextTheme);
});

tocToggle?.addEventListener("click", () => {
  toggleSidebar();
});

document.addEventListener("click", (event) => {
  if (
    window.innerWidth <= 960 &&
    noteSidebar.classList.contains("is-open") &&
    !noteSidebar.contains(event.target) &&
    !tocToggle.contains(event.target)
  ) {
    toggleSidebar(true);
  }
});

window.addEventListener("scroll", updateHeaderState, { passive: true });
window.addEventListener("resize", () => {
  if (window.innerWidth > 960) {
    toggleSidebar(true);
  }
});

setTheme(getPreferredTheme());
updateHeaderState();
loadNote();

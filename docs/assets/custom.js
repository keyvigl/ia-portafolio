/* ================================================================
   AI PORTFOLIO — custom.js
   Neural particles · Typing animation · Scroll reveal · Counters
   ================================================================ */

(function () {
  'use strict';

  /* ── 1. NEURAL NETWORK CANVAS ─────────────────────────────── */
  function initNeuralCanvas() {
    var canvas = document.getElementById('neural-canvas');
    if (!canvas) return;

    var ctx = canvas.getContext('2d');
    var particles = [];
    var mouse = { x: null, y: null };
    var animId;

    function resize() {
      canvas.width  = canvas.parentElement.offsetWidth;
      canvas.height = canvas.parentElement.offsetHeight;
    }

    function spawn() {
      particles = [];
      var count = Math.min(Math.floor((canvas.width * canvas.height) / 9000), 90);
      for (var i = 0; i < count; i++) {
        particles.push({
          x:  Math.random() * canvas.width,
          y:  Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * 0.45,
          vy: (Math.random() - 0.5) * 0.45,
          r:  Math.random() * 1.8 + 0.5,
          color: Math.random() > 0.5 ? '#00e5ff' : '#8b5cf6',
          opacity: Math.random() * 0.55 + 0.2
        });
      }
    }

    function step() {
      particles.forEach(function (p) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width)  p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        /* mouse repulsion */
        if (mouse.x !== null) {
          var dx = p.x - mouse.x;
          var dy = p.y - mouse.y;
          var d  = Math.sqrt(dx * dx + dy * dy);
          if (d < 110) {
            p.x += (dx / d) * 1.8;
            p.y += (dy / d) * 1.8;
          }
        }
      });
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      /* connections */
      for (var i = 0; i < particles.length; i++) {
        for (var j = i + 1; j < particles.length; j++) {
          var dx = particles[i].x - particles[j].x;
          var dy = particles[i].y - particles[j].y;
          var d  = Math.sqrt(dx * dx + dy * dy);
          if (d < 135) {
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(0,229,255,' + ((1 - d / 135) * 0.22) + ')';
            ctx.lineWidth   = 0.6;
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }

      /* nodes */
      particles.forEach(function (p) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle   = p.color;
        ctx.globalAlpha = p.opacity;
        ctx.fill();
        ctx.globalAlpha = 1;
      });
    }

    function loop() {
      step();
      draw();
      animId = requestAnimationFrame(loop);
    }

    resize();
    spawn();
    loop();

    window.addEventListener('resize', function () {
      cancelAnimationFrame(animId);
      resize();
      spawn();
      loop();
    });

    /* Track mouse over the whole document for a wider interaction zone */
    document.addEventListener('mousemove', function (e) {
      var rect  = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
    });

    document.addEventListener('mouseleave', function () {
      mouse.x = null;
      mouse.y = null;
    });
  }

  /* ── 2. TYPING ANIMATION ──────────────────────────────────── */
  function initTyping() {
    var el = document.getElementById('typing-target');
    if (!el) return;

    var texts   = ['Keyvi Garcia', 'Data Scientist', 'ML Engineer', 'AI Developer', 'Keyvi Garcia'];
    var tIdx    = 0;
    var cIdx    = 0;
    var deleting = false;

    function tick() {
      var current = texts[tIdx];
      if (deleting) {
        cIdx--;
        el.textContent = current.slice(0, cIdx);
      } else {
        cIdx++;
        el.textContent = current.slice(0, cIdx);
      }

      var delay = deleting ? 55 : 105;

      if (!deleting && cIdx === current.length) {
        delay    = 2400;
        deleting = true;
      } else if (deleting && cIdx === 0) {
        deleting = false;
        tIdx     = (tIdx + 1) % texts.length;
        delay    = 380;
      }

      setTimeout(tick, delay);
    }

    tick();
  }

  /* ── 3. SCROLL REVEAL ─────────────────────────────────────── */
  function initReveal() {
    var els = document.querySelectorAll('.reveal');
    if (!els.length || !window.IntersectionObserver) return;

    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) e.target.classList.add('visible');
      });
    }, { threshold: 0.1 });

    els.forEach(function (el) { obs.observe(el); });
  }

  /* ── 4. ANIMATED COUNTERS ─────────────────────────────────── */
  function initCounters() {
    var els = document.querySelectorAll('.stat-number[data-count]');
    if (!els.length || !window.IntersectionObserver) return;

    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        var el     = entry.target;
        var target = parseInt(el.getAttribute('data-count'), 10);
        var start  = performance.now();
        var dur    = 1800;

        function frame(now) {
          var prog = Math.min((now - start) / dur, 1);
          var ease = 1 - Math.pow(1 - prog, 3); /* cubic ease-out */
          el.textContent = Math.floor(ease * target);
          if (prog < 1) requestAnimationFrame(frame);
          else el.textContent = target;
        }

        requestAnimationFrame(frame);
        obs.unobserve(el);
      });
    }, { threshold: 0.5 });

    els.forEach(function (el) { obs.observe(el); });
  }

  /* ── 5. TECH TAG STAGGER ──────────────────────────────────── */
  function initTagStagger() {
    document.querySelectorAll('.tech-tag').forEach(function (tag, i) {
      tag.style.opacity  = '0';
      tag.style.animation =
        'fadeInUp 0.5s ease ' + (0.65 + i * 0.07) + 's forwards';
    });
  }

  /* ── 6. CURSOR GLOW (subtle) ──────────────────────────────── */
  function initCursorGlow() {
    var glow = document.createElement('div');
    glow.style.cssText = [
      'position:fixed',
      'pointer-events:none',
      'z-index:9999',
      'width:320px',
      'height:320px',
      'border-radius:50%',
      'transform:translate(-50%,-50%)',
      'background:radial-gradient(circle,rgba(0,229,255,.06) 0%,transparent 65%)',
      'transition:left .12s ease,top .12s ease',
      'left:-999px',
      'top:-999px'
    ].join(';');
    document.body.appendChild(glow);

    document.addEventListener('mousemove', function (e) {
      glow.style.left = e.clientX + 'px';
      glow.style.top  = e.clientY + 'px';
    });
  }

  /* ── INIT ─────────────────────────────────────────────────── */
  function init() {
    initNeuralCanvas();
    initTyping();
    initReveal();
    initCounters();
    initTagStagger();
    initCursorGlow();
  }

  /* Material instant-navigation awareness */
  if (typeof document$ !== 'undefined') {
    document$.subscribe(init);
  } else if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

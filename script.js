document.addEventListener('DOMContentLoaded', () => {
    // --- Filtering Logic ---
    const filterBtns = document.querySelectorAll('.nav-btn');
    const cards = document.querySelectorAll('.card');

    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all buttons
            filterBtns.forEach(b => b.classList.remove('active'));
            // Add active class to clicked button
            btn.classList.add('active');

            const filterValue = btn.getAttribute('data-filter');

            cards.forEach(card => {
                if (filterValue === 'all' || card.getAttribute('data-category') === filterValue) {
                    card.style.display = 'flex'; // Changed to flex for new layout
                } else {
                    card.style.display = 'none';
                }
            });
        });
    });

    // --- Tab Logic ---
    const tabBtns = document.querySelectorAll('.tab-btn');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const card = e.target.closest('.card');
            const targetView = e.target.getAttribute('data-target');

            // Update Tab Buttons
            card.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');

            // Update View Content
            card.querySelectorAll('.view-content').forEach(content => {
                if (content.getAttribute('data-view') === targetView) {
                    content.classList.add('active');
                } else {
                    content.classList.remove('active');
                }
            });
        });
    });

    // --- Modal Logic ---
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImg');
    const closeBtn = document.getElementsByClassName('close')[0];
    const viewBtns = document.querySelectorAll('.view-btn');

    viewBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            // Find the closest card-image container to the clicked button
            const cardImageContainer = e.target.closest('.card-image');
            const img = cardImageContainer.querySelector('img');

            modal.style.display = 'block';
            modalImg.src = img.src;
        });
    });

    closeBtn.onclick = function () {
        modal.style.display = 'none';
    }

    // Close modal when clicking outside the image
    window.onclick = function (event) {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    }

    // Close modal with Escape key
    document.addEventListener('keydown', function (event) {
        if (event.key === "Escape") {
            modal.style.display = 'none';
        }
    });
});

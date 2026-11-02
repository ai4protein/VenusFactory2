document.addEventListener('DOMContentLoaded', function() {
    // add click event to all navigation links
    document.querySelectorAll('.manual-nav a').forEach(function(link) {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // add click event to all images in the manual content
    document.querySelectorAll('.manual-content img').forEach(function(img) {
        img.addEventListener('click', function() {
            // create a modal box to display the large image
            const modal = document.createElement('div');
            modal.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; background-color:rgba(0,0,0,0.8); display:flex; justify-content:center; align-items:center; z-index:9999;';
            
            // create a large image element
            const largeImg = document.createElement('img');
            largeImg.src = this.src;
            largeImg.style.cssText = 'max-width:90%; max-height:90%; object-fit:contain;';
            
            // add the large image to the modal box
            modal.appendChild(largeImg);
            
            // click the modal box to close it
            modal.addEventListener('click', function() {
                document.body.removeChild(modal);
            });
            
            // add the modal box to the body
            document.body.appendChild(modal);
        });
    });
});
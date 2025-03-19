if ('serviceWorker' in navigator) {
    navigator.serviceWorker.getRegistrations().then(function(registrations) {
      for (let registration of registrations) {
        registration.unregister();
      }
    });
  }
  
  function refreshCacheAndReload() {
    fetch(location.href, { cache: "no-store" }).then(() => {
      location.reload(true);
    });
  }
  
  window.onload = function() {
    setTimeout(refreshCacheAndReload, 2000);
  };
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.type === 'entities') {
      const entitiesBox = document.getElementById('entity-text-box');
      entities = request.entities;
      entities.forEach((entity) => {
        console.log('adding ent');
        console.log(entity);
        entityTextBox.value += entity + '\n';
      });
    }
});

chrome.contextMenus.create({
  id: "NER",
  title: "Perform NER on this page",
  contexts: ["selection"]
});

chrome.contextMenus.onClicked.addListener(function(info){
  performNER(info.selectionText);
  chrome.tabs.sendMessage(tab.id, { text: 'invertColors' });
  console.log('end message');
});

function showEntitiesInPopup(entities, labels) {
  // Get the popup window and the entity text box
  
    const popupWindow = chrome.extension.getViews({ type: 'popup' })[0];
    const NameTextBox = popupWindow.document.getElementById('names-box');
    const DateTextBox = popupWindow.document.getElementById('dates-box');
    const LocTextBox = popupWindow.document.getElementById('locs-box');
    const OrgTextBox = popupWindow.document.getElementById('org-box');
    const OthTextBox = popupWindow.document.getElementById('other-box');
  // Clear any previous text in the entity text box
   // entityTextBox.value = '';
  
  // Add the entities to the text box
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] === "GPE") {
        LocTextBox.value += entities[i] + '\n';
      } else if (labels[i] === "PERSON") {
        NameTextBox.value += entities[i] + '\n';
      } else if (labels[i] === "DATE") {
        DateTextBox.value += entities[i] + '\n';
      } else if (labels[i] == "ORG") {
        OrgTextBox.value += entities[i] + '\n';
      } else {
        OthTextBox.value += entities[i] + '\n';
      }
    }

    console.log("done");
  }

function performNER(texts) {
    console.log("performingner");
    fetch('http://localhost:8000/ner/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ 'text': texts })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.ents);
        console.log(data.labels);
        showEntitiesInPopup(data.ents, data.labels);

        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
          chrome.tabs.sendMessage(tabs[0].id, {greeting: "hello", entz: data.ents, labz: data.labels}, function(response) {
            console.log(response.farewell);
          });
        });
    })
    .catch(error => console.error(error));
}


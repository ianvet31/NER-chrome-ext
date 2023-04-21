chrome.runtime.onMessage.addListener(
  function(request, sender, sendResponse) {
    console.log(sender.tab ?
                "from a content script:" + sender.tab.url :
                "from the extension");
    if (request.greeting == "hello")
      sendResponse({farewell: "goodbye"});
      console.log(request.labz);
      console.log(request.entz);
      highlighter(request.entz, request.labz);
  }
);


function highlighter(ents, labels) {

  let replacement = '<span style="background-color: yellow;">$&</span>';
  let regex = /\bObama\b/gi;
  
  for (let i = 0; i < labels.length; i++) {
    
    if (labels[i] == "PERSON") {
      replacement = '<span style="background-color: yellow;">$&</span>';
    } else if (labels[i] == "DATE") {
      replacement = '<span style="background-color: red;">$&</span>';
    } else if (labels[i] == "GPE") {
      replacement = '<span style="background-color: blue;">$&</span>';
    } else if (labels[i] == "ORG") {
      replacement = '<span style="background-color: green;">$&</span>';
    } else {
      replacement = '<span style="background-color: orange;">$&</span>';
    }
    regex = new RegExp(`\\b${ents[i]}\\b`, 'gi');
    document.body.innerHTML = document.body.innerHTML.replace(regex, replacement);
  }
    
}


/*

function highlighter(ents, labels) {
  /*
  const regex = /\bObama\b/gi;
  const replacement = '<span style="background-color: yellow;">$&</span>';
  
  document.body.innerHTML = document.body.innerHTML.replace(regex, replacement);

  let replacement = '<span style="background-color: yellow;">$&</span>';
  let regex = /\bObama\b/gi;
  
  for (let i = 0; i < labels.length; i++) {
    
    if (labels[i] == "PERSON") {
      replacement = '<span style="background-color: yellow;">$&</span>';
    } else if (labels[i] == "DATE") {
      replacement = '<span style="background-color: red;">$&</span>';
    } else if (labels[i] == "GPE") {
      replacement = '<span style="background-color: blue;">$&</span>';
    } else if (labels[i] == "ORG") {
      replacement = '<span style="background-color: green;">$&</span>';
    } else {
      replacement = '<span style="background-color: orange;">$&</span>';
    }
    regex = new RegExp(`\\b${ents[i]}\\b`, 'gi');
    document.body.innerHTML = document.body.innerHTML.replace(regex, replacement);
  }
    
}
*/



// Listen for a message from the background script to initiate highlighting
/*
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  if (request.type === "entities") {
    console.log('i hear ya');
    highlightEntities(request.entities);
    underlineStringsOnPage(request.entities);
  }
});

*/

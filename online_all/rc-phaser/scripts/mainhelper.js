const serverBaseURL = "http://localhost:8888"; // URL to the project located on the server, no trailing slash!
// https://hg-lm-s001.econ.uzh.ch/staging

// function to format the date and time
function formatDate(date) {
  var year = date.getFullYear();
  var month = date.getMonth() + 1; // months are zero indexed
  month = month < 10 ? "0" + month : month;
  var day = date.getDate();
  var hour = date.getHours();
  var minute = date.getMinutes();
  minute = minute < 10 ? "0" + minute : minute;
  var second = date.getSeconds();
  second = second < 10 ? "0" + second : second;
  return (
    day + "-" + month + "-" + year + "_" + hour + "-" + minute + "-" + second
  );
  }

function getTime() {
  //make a new date object
  let d = new Date();
  //return the number of milliseconds since 1 January 1970 00:00:00.
  return d.getTime();
}

 // ----------------- Visualization ---------------------------
// add fullscreen button
function fullscreen(scene) {
  if (scene.scale.isFullscreen)    {
    var button = scene.add.image(1280-16, 16, 'fullscreen', 1).setOrigin(1, 0).setInteractive();
  } else {
    var button = scene.add.image(1280-16, 16, 'fullscreen', 0).setOrigin(1, 0).setInteractive();
  }

  button.on('pointerup', function () {
    var time = new Date();
    time = formatDate(time);
    
    if (scene.scale.isFullscreen)    {
      button.setFrame(0);
      scene.scale.stopFullscreen();
      scene.registry.values.fullscreenEnd.push(time);
    } else {
      button.setFrame(1);
      scene.scale.startFullscreen();
      scene.registry.values.fullscreenStart.push(time); }
    }, scene);
  }
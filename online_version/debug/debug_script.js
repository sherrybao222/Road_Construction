
var date = new Date();
document.body.innerHTML = "<h1>Today is: " + date + "</h1>";
// this is a comment
/* this is a multiple
line comments */

//combine two list
let a = [2,3,5];
let b = [5,2,1,2,4,5];
let c = a.map(x => [x,b[a.indexOf(x)]]);
// console.log(c);

//this should work fine in a server not not get local files
// fetch("./basic_map_training.json")
//   .then(function(resp) {
//     return resp.json();
//   })
//   .then(function(data) {
//     consolge.log(data);
//   })

// let json = $.getJSON("basic_map_training.json", function(json) {
//     console.log(json); // this will show the info it in firebug console
// });

let city = [10,20,30,0,4];
let city_2 = city.filter(i => i<20 && i != 0);
// var city_2 = [for (x of city) if (x<20 && x != 0) x];
// console.log(city_2);

let d = [];
// console.log(d.length);

for (i of a){
  b[i]=0;
};

//create new array and fill in array
let apple = Array.from(Array(12).keys());
let incentive_score = [];
for (let i of apple){
  i = (i**2) * 0.01;
  incentive_score.push(i);
};
// console.log(incentive_score);

//remove the last item from array > working GOOD
//console.log(city);
//city.splice(city.length-1,1)
//console.log(city);

//saving json try
const fs = require('fs');
//require is server side
const map_data = {
  id: 1,
  cities: 30,
  undo: 1,
}

const jsonString = JSON.stringify(map_data);
//stringify json object to string
//console.log(jsonString);
//then write stringfied to file
fs.writeFile("output.json", jsonContent, 'utf8', function (err) {
    if (err) {
        console.log("An error occured while writing JSON Object to File.");
        return console.log(err);
    }
    console.log("JSON file has been saved.");
});

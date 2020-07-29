//this is the command that load maps using server
//need to load this file and map file in html

fetch("./basic_map_training.json")
.then(function(resp){
return resp.json();
})
.then(function(data){
console.log(data)
});

async function getProfile(accessToken) {
    let accessToken = localStorage.getItem('access_token');
  
    const response = await fetch('https://api.spotify.com/v1/me', {
      headers: {
        Authorization: 'Bearer ' + accessToken
      }
    });
  
    const data = await response.json();
  }


const APIController = (function myAPI(){
    const clientID = "4d95d55f4baa48bfa1b250b737e14b74";

    console.log(clientID);
});


let x = "node ran successfully";
console.log(x);
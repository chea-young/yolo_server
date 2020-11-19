
importScripts('https://www.gstatic.com/firebasejs/4.8.1/firebase-app.js');
importScripts('https://www.gstatic.com/firebasejs/4.8.1/firebase-messaging.js');
 
// Initialize Firebase
var config = {
  apiKey: "AIzaSyBfLYQieTSNFJVghLmwPidH9eMLJ8sgafA",
    authDomain: "push-server-56a52.firebaseapp.com",
    databaseURL: "https://push-server-56a52.firebaseio.com",
    projectId: "push-server-56a52",
    storageBucket: "push-server-56a52.appspot.com",
    messagingSenderId: "967213415011",
    appId: "1:967213415011:web:a14d673747dd9f6a2dd656",
    measurementId: "G-RN5H6RK362"
};
firebase.initializeApp(config);
 
const messaging = firebase.messaging();
messaging.setBackgroundMessageHandler(function(payload){ 
    const title = payload.data.title;
    const options = {
           body: payload.data.body
    };
    console.log(payload)
    //raise_norification(payload.data.title, payload.data.body)
    return self.registration.showNotification(title,options);
});
//messaging.onMessage((payload) => {
 // console.log('[firebase-messaging-sw.js] Received background message ', payload);
  //var notificationTitle = payload.data.title;
  //var notificationOptions = {
   // body: payload.data.body,
 // };
 // console.log(payload.data)
 // return self.registration.showNotification(notificationTitle,
 //   notificationOptions);
//});
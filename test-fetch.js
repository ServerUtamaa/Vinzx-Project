const https = require('https');
https.get('https://nfs.faireconomy.media/ff_calendar_nextweek.json', (res) => {
  console.log('nextweek status:', res.statusCode);
});
https.get('https://nfs.faireconomy.media/ff_calendar_thisweek.json', (res) => {
  console.log('thisweek status:', res.statusCode);
});

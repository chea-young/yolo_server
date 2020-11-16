from firebase_admin import messaging

def send_to_firebase_cloud_messaging():
    # This registration token comes from the client FCM SDKs.
    registration_token = 'cHXdiCBkUFM:APA91bFI01-x0KnqCSSJRCh7iD-50rprDalwsom5nhdcHCDomm9XLc7m9rJAR-OsRJzFLJ-YctQUsfTs6um_wO4yb476s6b_frfVPom94_CwJoo7JwKG1iOdbmBg4MrmV-PwCQOTLUC-'

    # See documentation on defining a message payload.
    message = messaging.Message(
    notification=messaging.Notification(
        title='안녕하세요 타이틀 입니다',
        body='안녕하세요',
        image='/Users/u_rim/Desktop/aa.png' 
    ),
    token=registration_token,
    data={'case':'accidnet', 'cctv_id':'서울역'},
    )

    response = messaging.send(message)
    # Response is a message ID string.
    print('Successfully sent message:', response)

def send_to_token():
    #default_app = firebase_admin.initialize_app()
    registration_token = 'cHXdiCBkUFM:APA91bFI01-x0KnqCSSJRCh7iD-50rprDalwsom5nhdcHCDomm9XLc7m9rJAR-OsRJzFLJ-YctQUsfTs6um_wO4yb476s6b_frfVPom94_CwJoo7JwKG1iOdbmBg4MrmV-PwCQOTLUC-'
    message = messaging.Message(
        data={
            'title':'안녕하세요 ',
            'body':'안녕',
            'largeIcon':'/Users/u_rim/Desktop/aa.png' 
        },
        token=registration_token,
    )
    response = messaging.send(message)
    # Response is a message ID string.
    print('Successfully sent message:', response)
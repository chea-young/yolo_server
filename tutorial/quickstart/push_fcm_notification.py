from firebase_admin import messaging

def send_to_firebase_cloud_messaging():
    # This registration token comes from the client FCM SDKs.
    registration_token = 'ctXWQUgsJ9k:APA91bE9wy-FiiFjja9oxV0o27Az48hKGUkxAICdoKzS5MtACYfuOGuSaPAlZ0uuo0I5Ay70v7SgiWIHCLvAPNMjVvnZUYRKiIfBRn6i8DxIsJ8pz9lYz8q5i1jxeh8cBqR7cpnAxp6F'

    # See documentation on defining a message payload.
    message = messaging.Message(
    notification=messaging.Notification(
        title='안녕하세요 타이틀 입니다',
        body='안녕하세요',    
    ),
    token=registration_token,
    data={'case':'accidnet', 'cctv_id':'서울역'},
    )

    response = messaging.send(message)
    # Response is a message ID string.
    print('Successfully sent message:', response)
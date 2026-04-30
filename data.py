TOPIC_DEFS = {
    "Course organization and structure": "The feedback explicitly mentions course organization or structure.",
    "Pace": "The feedback explicitly mentions the pace of the course.",
    "Workload": "The feedback explicitly mentions the courses workload.",
    "Student engagement and participation": "The feedback explicitly mentions student engagement and participation in the course.",
    "Clarity of explanations": "The feedback explicitly states how the instructor explained concepts.",
    "Effectiveness of assignments": "The feedback explicitly mentions the effectiveness of assignments.",
    "Classroom atmosphere": "The feedback explicitly mentions if the class atmosphere was supportive or not.",
    "Instructor's communication and availability": "The feedback explicitly mentions the instructors accessibility and availability.",
    "Inclusivity and sense of belonging": "The feedback explicitly mentions feeling included by the instructor in class.",
    "Assessment": "The feedback explicitly mentions the alignment of assessments with course material.",
    "Grading and feedback": "The feedback explicitly mentions grading fairness, or feedback received on performance and assignments.",
    "Learning resources and materials": "The feedback explicitly mentions how useful or bad the learning materials provided were."
}

TOPIC_KEYS = [t for t in TOPIC_DEFS.keys()]

FEEDBACK_LIST = [
    "Exactly as the class should have been. Professor Wu obviously cares about his students success and I feel as though he did everything possible to set us up for success. I don't know if this will come across the way I intend it to but I really appreciate how normal this class was. What I mean is that it didn't consume too much of my time or attention but I still came out having learned everything I needed to; it was a very seamless experience. I feel as though I got exactly what I wanted out of this class (a stress–free experience learning the basics of chemistry) and had I wanted something else (say, an introduction to the marvels of chemistry) Wu could have provided that too. Thank you professor Wu and keep up the good work <3",
    
    "Eric Wu is a HUGE step up from the last professor who taught chemistry. He is a phenomenal educator who focuses mostly on the logical and understanding parts of chemistry. Taking out most of the memorizing parts, it allows students to fully immerse into thinking how chemistry works. Afterall, it is how the exams, homework, and discussions are structured, solving through deconstruction and reconstruction of the problem. The lectures themselves explains thoroughly through the steps of each step, calculations, or concept. Lectures are beautifully structured with multiple options to engage such as asking questions during the lecture through a QR code, provided downloadable notes, office hours, and plenty more.",
    
    "Excellent Professor. Such a shame he isn't teaching OChem next quarter",
    
    "Great professor, the only thing was that the lectures were very fast. Sometimes it made it hard to follow along. I think a good solution would be to do less examples, and spend more times on the ones that we do in lecture.",
    
    "Professor Wu is a wonderful teacher and is very organized and clear with his teaching. I appreciate that he provided a myriad of resources and practice material along with lecture notes and midterm review sessions. He explains well the content of the class and makes frequent pauses throughout his lectures for answering questions. I appreciate that he is a fair grader and provides opportunities for students to achieve the best grade possible in the class. The pace of the class did seem a bit fast and rushed at times, but overall, Professor Wu did a good job of covering the material well in the ten weeks of the quarter.",
    
    "Dr. Wu's knowledge of chemistry is evident in his teaching. Every lecture was well prepared and provided information in a clear and organized way. It is also obvious that Dr. Wu truly cares about helping students learn. He provides a lot of helpful resources before every exam and also holds review sessions. Overall, he is a great teacher.",
    
    "One of Wu's greatest strengths is his lecture–filled notes. If we are already familiar with the course material, we can just follow the lecture with the notes without having to write down anything. It is also a great tool for reviewing before, say, a midterm or the final exam. Another great strength of his is his coffee hours that he has with students after class. It's nice to get to know your teacher personally and establish a great connection. His final strength is the review sessions he has before midterms. They are well–organized and go over a good chunk of the material. The only weakness I can think of with Wu is how fast he goes during his lectures.",
    
    "Incredible teacher that was fair and taught me a lot!",
    
    "Eric explains concepts very clearly with an emphasis on intuition and he encourages students to understand the material rather than rely solely on rote memory. His enthusiasm for chemistry is contagious. He is extremely welcoming, knowledgeable, approachable, and responsive to questions, and he takes the time to make sure students understand what is going on.",
    
    "Very excellent professor and lecturer. The environment he fosters for his students is impeccable, and his dedication towards his students shines.",
    
    "Very good at lecturing and very good at giving extra materials to study. Tests and homework were very fair and tested what we learned in class. One of the best professors I have had at UCLA in 5 quarters.",
    
    "Best professor ever",
    
    "Professor Wu is extremely understanding and caring about his students needs. During the fires he was the only professor of mine to offer accommodations to those struggling. Not only is he extremely kind but he is a very good lecturer as well.",
    
    "Professor Wu was very organized with his lecture notes and always posted materials online, which were very helpful. His review sessions prior to midterms were also very helpful and definitely contributed to my success in the class.",
    
    "Professor Wu is very good at explaining math concepts, but he doesn't always explain the conceptual side of things very well. I think that he could focus a bit more on explaining the conceptual aspects of the class.",
    
    "Very organized, very manageable workload and difficulty, extremely helpful resources and practice problems for studying, lectures moved a little fast at times but not too bad, very conducive to learning overall",
    
    "Great lecture style and posts notes and recordings which is very useful. Test prep is provided which I like. No complaints!",
    
    "Fair with his exams and an excellent lecturer",
    
    "Professor Wu is one of the best professors I've had since coming to UCLA. I really like his teaching style and how much he cares for his students. His lectures are consistently clear and paced well. His exams are fair – not too challenging, but also not too easy.",
    
    "Professor Wu is very organized and teaches in a way that was digestible. I hope my future teachers are like him because even though this class was tough, I was able to thrive due to his teaching techniques.",
    
    "Excellent communicator and very organized. Only weakness of this course is that it seems to do a lot of hand–waving over more complex concepts.",
    
    "All the info needed for exams were on his slides and given to us. I think the exams tested our knowledge adequately. I think the pace of his class was quick which made me overwhelmed at times.",
    
    "He was the best professor I have had so far. He takes the time to really help us grasp the topic while still keeping us at a good pace to finish the class.",
    
    "I really enjoyed being in Professor Wu's class this quarter! I would highly recommend anyone to take him in the future. He gives clear explanations for topics and provides lots of examples.",
    
    "Instructor E.C. Wu demonstrates several strengths in his teaching, including kindness, understanding, and a genuine care for his students.",
    
    "Professor Wu is great!",
    
    "Professor Wu is a fantastic professor all around. He accounts for different learning styles and is very thorough in his lectures.",
    
    "Great at teaching the content and makes the learning very straightforward. Gives wonderful examples to help his students learn.",
    
    "I was already quite knowledgeable about the course material before taking this course, but I still found professor Wu to be an incredibly dedicated and knowledgeable professor.",
    
    "Professor Wu was very good about making students feel welcome and he made himself very approachable.",
    
    "He's awesome, does lecture in a clear and concise way, explains what we need to know before exams, class is organized.",
    
    "Professor Wu led an organized, well structured class with a lot of helpful resources.",
    
    "Dr. Wu is very organized and his lectures reflect his test and quizzes.",
    
    "Wu is very explanatory and in depth with the material. However, material is very advanced for those without a background in chemistry.",
    
    "Great lecturer and note taker, though tends to talk too quickly.",
    
    "He was an amazing professor who provided us with so much resources and truly wanted us to succeed.",
    
    "Eric is an excellent professor. From his lectures to his homework to his tests he is very organized and very communicative.",
    
    "Professor Wu is extremely organized and explains concepts very clearly. He truly wants to help students succeed.",
    
    "Chemistry 20B can be challenging at times but Professor Wu was probably the best person for the job.",
    
    "I think that Dr. Wu was an excellent professor and one of the best professors I have had so far.",
    
    "Eric Wu is the best professor I have had so far at UCLA.",
    
    "Excellent course. Only criticism is lectures move very quickly.",
    
    "He has no weaknesses, his main strength is the way he can teach.",
    
    "Professor Wu understands how to teach an undergraduate course. Workloads are manageable and the class is well–taught.",
    
    "Phenomenal professor, explains everything thoroughly in lectures and prepares perfect practice and review sessions.",
    
    "Professor Eric was always prepared with clear lectures and notes, and provided us with various resources.",
    
    "I like his teaching and the problem set problems are effective in teaching me the material.",
    
    "Professor Wu was an excellent teacher. He explained topics well, and I really appreciate that he recorded the lectures.",
    
    "Eric Wu is my goat.",
    
    "Dr. Wu was amazing! Great lecturer who is very well organized and clearly cares about his students.",
    
    "Professor Wu is an amazing professor! His lectures are super organized and he gives students notetakers as a guide.",
    
    "Professor Wu is very approachable and seems to care about his students.",
    
    "Chemistry is difficult and a lot to digest, but Professor Wu made it extremely manageable.",
    
    "Dr. Wu genuinely cares for his students and wants us to succeed.",
    
    "I think Dr. Wu is always very helpful. He is really good at explaining topics in a way that is easy to understand.",
    
    "Dr. Wu showed a lot of concern for the students and was always making sure to answer any confusion.",
    
    "He is one of the best professors I've had so far, his explanations are so clear.",
    
    "Eric Wu cares about his students and provides ample support and resources.",
    
    "Very organized, clear and concise.",
    
    "The professor clearly cares about student learning and provides plenty of helpful material.",
    
    "His ability to explain concepts in a way that keeps students engaged.",
    
    "Eric Wu does a very good job of lecturing and making sure we understand the concepts.",
    
    "Professor Wu was very organized in his lecture notes and very good at breaking down complicated concepts.",
    
    "Professor Wu is thorough and complete in his teaching.",
    
    "Instructor was communicative, approachable, and very concerned with student success.",
    
    "Strengths: simplifying course material. Weakness: fast pacing.",
    
    "Dr. Wu hosts helpful review sessions and makes lectures entertaining.",
    
    "He is a great lecturer but his midterms are too long for 50 minutes.",
    
    "Everything helped me learn, and no time was wasted.",
    
    "Professor Wu is an amazing professor who cares deeply about his students.",
    
    "This instructor was very caring and knowledgeable.",
    
    "Great job, I love this class!",
    
    "Professor Wu knows what he's teaching and can be understood by many.",
    
    "Eric helped to make Chemistry approachable.",
    
    "Professor Wu is extremely organized and adapts based on feedback.",
    
    "Very organized and great communication.",
    
    "Good communicator and always engaged with students.",
    
    "Some strengths include organization and being welcoming.",
    
    "I really like the notes for this course.",
    
    "Prof Wu is the GOAT.",
    
    "Eric Wu is one of the nicest professors I have ever had.",
    
    "Eric is such a nice and empathetic person.",
    
    "Very good professor and very interesting.",
    
    "Professor Wu is energetic and capable.",
    
    "Professor Wu was so helpful and passionate.",
    
    "Professor Wu is extremely organized and clear as a lecturer.",
    
    "Keep doing what you're doing, cause everyone loves it!",
    
    "I loved Eric Wu. He is such an enthusiastic professor.",
    
    "Professor Wu is truly an outstanding educator!",
    
    "Professor Wu goes above and beyond for students.",
    
    "Overall, Professor Wu was an amazing instructor.",
    
    "Professor Wu is an amazing chemistry teacher!",
    
    "Super great lecturer!",
    
    "Eric Wu is the best lecturer I have ever had.",
    
    "Very organized, and the class material is easy to understand.",
    
    "Prof Wu is an excellent lecturer and instructor.",
    
    "This instructor is fantastic and very organized.",
    
    "Dr. Wu genuinely cares about students' learning.",
    
    "Eric is a very knowledgeable professor.",
    
    "Professor Wu is a talented professor.",
    
    "Professor Wu is one of the best professors I've had at UCLA.",
    
    "Eric is the most wonderful professor I have had at UCLA so far."
]

SCORING_RUBRIC = {
    "Course organization and structure": {
        1: "The course organization and structure makes it hard to understand/learn.",
        2: "The course organization and structure has several issues and can be confusing at times.",
        3: "The course organization and structure was okay.",
        4: "The course organization and structure was clear and worked well.",
        5: "The course organization and structure was excellent and made the material easy to understand."
    },

    "Pace": {
        1: "The course pace moves way too fast, making it hard to keep up.",
        2: "The course pace was far too slow and made it hard to stay engaged.",
        3: "The course pace moves somewhat fast.",
        4: "The course pace was somewhat slow at times.",
        5: "The course pace was reasonable overall."
    },

    "Workload": {
        1: "The course workload is too much; it is a struggle to keep up.",
        2: "The workload was extremely light and did not feel meaningful.",
        3: "The course workload feels a bit too much.",
        4: "The course workload was a bit light and could have been more challenging.",
        5: "The course workload was manageable."
    },

    "Student engagement and participation": {
        1: "The course had no meaningful opportunities for students to engage.",
        2: "There were limited opportunities to engage, and it was hard to stay involved at times.",
        3: "The level of engagement was okay overall.",
        4: "The course provided good opportunities to stay engaged.",
        5: "The course was highly engaging and consistently kept students engaged."
    },

    "Clarity of explanations": {
        1: "Explanations did not make sense, were vague, and made it difficult to understand.",
        2: "Explanations were not always clear, making some parts hard to understand.",
        3: "Explanations were somewhat clear and helped with basic understanding, but required additional effort to fully grasp.",
        4: "Explanations were clear and generally easy to understand.",
        5: "Explanations were very clear, and no additional time was needed to understand."
    },

    "Effectiveness of assignments": {
        1: "Assignments had no impact on understanding or did not help with practice.",
        2: "Assignments were barely helpful and had little impact on understanding.",
        3: "Assignments were somewhat helpful and supported basic understanding.",
        4: "The assignments were helpful and supported understanding of the material.",
        5: "Assignments were very effective and greatly improved understanding and practice."
    },

    "Classroom atmosphere": {
        1: "The class atmosphere is unwelcoming and demotivating.",
        2: "The class atmosphere is somewhat uncomfortable and makes it harder to succeed.",
        3: "The class atmosphere is okay overall.",
        4: "The class atmosphere is welcoming and supports learning.",
        5: "The class atmosphere is very welcoming and creates a positive learning environment."
    },

    "Instructor's communication and availability": {
        1: "The instructor rarely or never responded and did not keep the class informed or was not available much.",
        2: "The instructor was slow or inconsistent in responding and did not always keep the class updated or had limited availability.",
        3: "The instructor communicated adequately and kept the class reasonably informed or the instructor was available at times.",
        4: "The instructor communicated clearly and responded in a timely manner or the instructor was available.",
        5: "The instructor communicated very clearly, responded promptly, and consistently kept the class well-informed or the instructor was highly available."
    },

    "Inclusivity and sense of belonging": {
        1: "The course environment felt exclusionary and unwelcoming to certain students.",
        2: "The course was somewhat inclusive, but some students may have felt left out or overlooked.",
        3: "The course was generally inclusive, but nothing stood out.",
        4: "The course was inclusive and respectful of different students.",
        5: "The course was highly inclusive and created a welcoming environment for all students."
    },

    "Assessment": {
        1: "Assessments were unclear, unfair, or did not reflect course content.",
        2: "Assessments had some issues with clarity or alignment with the material.",
        3: "Assessments were acceptable and generally reflected the material.",
        4: "Assessments were clear and aligned with the course material.",
        5: "Assessments were very clear, fair, and well-aligned with course objectives."
    },

    "Grading and feedback": {
        1: "Grading was inconsistent, or feedback was not provided or was not helpful.",
        2: "Feedback was limited, unclear, or not very helpful.",
        3: "Feedback was provided, but not especially helpful or detailed.",
        4: "Feedback was helpful and supported understanding.",
        5: "Feedback was detailed, timely, and significantly improved learning."
    },

    "Learning resources and materials": {
        1: "Learning materials were missing, unclear, or not useful.",
        2: "Learning materials were somewhat helpful but had noticeable issues or gaps.",
        3: "Learning materials were adequate and supported basic understanding.",
        4: "Learning materials were helpful and supported the understanding of the course.",
        5: "Learning materials were very helpful, clear, and enhanced learning significantly."
    }
}

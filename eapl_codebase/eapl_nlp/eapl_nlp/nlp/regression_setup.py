try:
    from .nlp_glob import nlp_func_map
    from .nlp_ops import nlp_ops_funcs
    from .nlp_utils import *
    from .cfg_templates import nlp_cfg_templates
except ImportError:
    from nlp_glob import nlp_func_map
    from nlp_ops import nlp_ops_funcs
    from nlp_utils import *
    from cfg_templates import nlp_cfg_templates

nlp_regression_test = logging.getLogger(__name__)

data_templates_dict = {
    "data_d1": {
        "text_obj": [
            {
                "txt": "Simpplr is announcing a new Work From Home 2020 Communication Plan to support organizations with a guaranteed two-week deployment for its enterprise-wide internal communications platform. Simpplr built its software to help remote and distributed organizations connect and align their workforces by streamlining internal communications and connecting employees across disparate locations.Many organizations are focused on collaboration tools such as instant messaging and videoconferencing to bring employees together, but organizations should understand that these technologies alone are unlikely to unify the entire organization.\u201cEmployees need a centralized, single source-of-truth for centralized company communications. Leaders need a platform to lead the organization and reinforce employee morale to mitigate fear and doubt,\u201d said Dhiraj Sharma, CEO of Simpplr.Internal communications platforms like Simpplr can help for the following reasons:Employees need a trusted, clutter-free place for critical company communications and information: The news is constantly changing, so companies should establish a source-of-truth where critical news won\u2019t get lost (such as in messaging apps and email). The platform can become a company newsroom where updates are placed for clarity and alignment and to quash rumors.Prolonged employee isolation will require proactive work to keep the full organization connected: Employees need to feel connected to their company, not just the co-workers they have meetings with each day. This connection impacts employee engagement and helps remote employees maintain their personal networks. Thus, companies should consider investing in tools and processes that keep employees linked to the broader organization.Remote employees will need good news and emotional support: Working remotely for extended periods can make employees feel lonely and disengaged from the organization. By using an internal communications platform, leaders and internal communications teams can reinforce values, share success stories, and keep morale positive.As part of the Work From Home 2020 Communication Plan, Simpplr is promising companies a two-week deployment time frame and an accelerated onboarding package. Many Simpplr customers have already expanded their licenses this week to communicate across all of their remote employees and contractors."
            },
            {
                "txt": "So what\u2019s the best technology or solution for a remote workforce? Often overlooked, it\u2019s a platform that can streamline top-down critical information while aligning and connecting employees organization-wide: the modern intranet. A modern intranet or an internal communication platform fills in the gap that the most-often talked about enterprise technologies miss. While productivity software is designed for employee-to-employee communication and real-time collaboration, the intranet is purposefully designed to streamline critical top-down communications especially for a remote workforce. It\u2019s important to make the distinction between traditional intranets and modern intranets. Modern intranets are different from the intranets of the past. Among many other differentiators, modern intranets can alert and deliver instant updates to employees from anywhere, without relying on the help of IT. And in times like this, it\u2019s absolutely critical to be able to reach remote employees instantly.Having multiple kinds of technology is key to empower employees to work efficiently, but we must recognize how to appropriately use each remote work tool to not only keep employees productive but to preserve company culture and connection.Productivity and collaboration technologies do not address the human need to feel engaged and aligned with an organization. When your entire workforce is working remotely, it\u2019s inevitable that employees will begin to feel uncertain about where they stand. To make matters worse, prolonged isolation leads to loneliness and can affect an employee\u2019s self-esteem and sense of motivation. Employees are looking up to leaders for guidance, clarity, and direction, not the timeline of the next product launch."
            },
            {
                "txt": "Monday was the first mandatory work from home day that was put in place to ensure employee health and safety as we deal with an unexpected (or maybe some would say expected) pandemic. The day started off with a new daily check-in, bright and early at 8:30am (Thanks Sam) to brainstorm how to best communicate as we adjust to this new system for our team.I had to dust off the in-home office, remove anything and everything from eye line of the webcam that will now be frequently used for virtual meetings, and hide away from my cat’s meows since he thinks it’s a newfound playtime. While working from home has its benefits, good luck to those working parents with kids at home, I’m fortunate enough to not have a cat that knows how to open doors (yet).As a majority of the workforce transitions to a newly mandated work from home/remote work policy during this time, we need to adapt to the new way the workforce will function. At this time, companies are not only thinking about the health of employees and communities but how that will affect the way we work together to help reduce the risk of getting sick. Shifting to a stronger, if not 100%, remote work policy can be an adjustment for organizations that are used to operating within their physical office locations. Few organizations are prepared but you can take the necessary steps to make the experience not only productive but impactful for your employees. Here’s how our team is transitioning to being a remote team instead of sitting besides each other (literally, I sit in front of my manager)."
            },
            {
                "txt": "Remote collaboration technology has helped this mandatory work from home policy become easier. Remote working tools like video conferencing, chat and collaborative tools have made it possible to be productive and simulate a working environment while at home.While working from home does have its benefits, it also comes with it’s own fair share of obstacles. The most common one is keeping virtual employees engaged and up to date on communications. Without the relationship-building that comes with an in-person office setting, teams can lose their culture of collaboration. As virtual workplaces have expanded, workers reported that “their work lives often lacked a sense of community and the richness of collaboration. Many experienced far too little unstructured social contact.” To ensure ongoing success with this model, keeping remote employees connected is the No. 1 priority for managers. If you’re used to seeing a teammate 5 days of the week and then don’t see them at all, it can feel a little isolating. Having regularly scheduled team video calls are a great way to align the whole team and offer an opportunity to just catch up as if you were in the office. Even an ‘after hours virtual happy hour’ gives a chance for remote employees to feel engaged and connected to the broader organization. Instant messaging in the workplace is a great way to get fast, casual responses in an immediate manner instead of sending an email. It’s a great way to facilitate conversation with employees that are spread out geographically and closes the distance gap. Make sure that you’re still checking in on employees and facilitating whatever they may need."
            },
            {
                "txt": "Monday was the first mandatory work from home day that was put in place to ensure employee health and safety as we deal with an unexpected (or maybe some would say expected) pandemic. The day started off with a new daily check-in, bright and early at 8:30am (Thanks Sam) to brainstorm how to best communicate as we adjust to this new system for our team.I had to dust off the in-home office, remove anything and everything from eye line of the webcam that will now be frequently used for virtual meetings, and hide away from my cat’s meows since he thinks it’s a newfound playtime. While working from home has its benefits, good luck to those working parents with kids at home, I’m fortunate enough to not have a cat that knows how to open doors (yet).As a majority of the workforce transitions to a newly mandated work from home/remote work policy during this time, we need to adapt to the new way the workforce will function. At this time, companies are not only thinking about the health of employees and communities but how that will affect the way we work together to help reduce the risk of getting sick. Shifting to a stronger, if not 100%, remote work policy can be an adjustment for organizations that are used to operating within their physical office locations. Few organizations are prepared but you can take the necessary steps to make the experience not only productive but impactful for your employees. Here’s how our team is transitioning to being a remote team instead of sitting besides each other (literally, I sit in front of my manager)."
            }
        ]

    },
    "data_d2": {
        "text_obj": [
            {
                "txt": "what is my name?",
                "label": ["Question", "Personal"]
            },
            {
                "txt": "what is your name?",
                "label": ["Question", "Personal"]
            },
            {
                "txt": "My name is Vishwas.",
                "label": ["Statement"]
            }
        ],
        "test_obj": [
            {
                "body": "Where is Bangalore?"
            },
            {
                "body": "Weather in Bangalore is good."
            }
        ],
        "input_lst_key": {
            "input_key": ["what is your name?", "what is my name?", "My name is vishwas", "I am from bangalore"],
            "label_key": ["Question", "Question", "Statement", "Statement"]
        }

    },
    "data_d3": {
        "text_obj": [
            {
                "txt": "How to assign alternate managers in SuccessFactors Learning?"
            },
            {
                "txt": "How to record costs for learning events in SuccessFactors Learning?"
            }
        ]

    },

    "data_d4": {
        'text_obj': [
            {
                'txt': "The quick brown fox jumps over the lazy dog.",
                'txt_lst': [
                    "The quick brown fox jumps over the lazy dog.",
                    "I am a sentence for which I would like to get its embedding",
                    "the fast brown fox jumps over the lazy dog"],

                'questions_rich': [
                    {"question": "The quick brown fox jumps over the lazy dog."},
                    {"question": "I am a sentence for which I would like to get its embedding"},
                    {"question": "the fast brown fox jumps over the lazy dog"}
                ],

                'drop_txt': [
                    {"txt": "the fast brown fox jumps over the lazy dog"}
                ]
            }
        ],
        'questions_rich': [
            {"question": "The quick brown fox jumps over the lazy dog."},
            {"question": "I am a sentence for which I would like to get its embedding"},
            {"question": "the fast brown fox jumps over the lazy dog"}
        ],

        'data_drop_txt': [
            {"txt": "the fast brown fox jumps over the lazy dog"}
        ],

        'data_drop_list': [
            "the fast brown fox jumps over the lazy dog",
            "some fox jumps over"
        ]
    },
    "data_d5": {
        "bot_name": "test",
        "query": "what is Select Statement?"
    },
    "data_d6": {
        'text_obj': [
            {
                'txt': "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."
            },
            {
                'txt': "Compatibility 2 of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."
            }
        ]
    }

}

reg_test_nlp_cfg = {
    'text_sum_nlp_cfg': {
        "cfg_seq_type": "nlp_cfg_template",
        "config_seq": ["text_summarization_init_cfg", "text_summarization_inference_config"]
    },
    'spacy_classifier_nlp_cfg': {
        "cfg_seq_type": "nlp_cfg_template",
        "config_seq": ["spacy_classifier"]
    },
    't5_base_nlp_cfg': {
        "cfg_seq_type": "nlp_cfg_template",
        "config_seq": ["t5_base_question_gen_init", "t5_base_question_gen_inference"]
    },
    't5_pipe_nlp_cfg': {
        "cfg_seq_type": "nlp_cfg_template",
        "config_seq": ["t5_pipe_question_gen_init", "t5_pipe_question_gen_inference"]
    },
    'alt_utt_nlp_cfg': {
        "cfg_seq_type": "nlp_cfg_template",
        "config_seq": ["alt_utt"]
    },
    'text_embed_nlp_cfg': {
        "cfg_seq_type": "nlp_cfg_template",
        "config_seq": ["text_embed"]
    },
    'hs_search_nlp_cfg': {
        "cfg_seq_type": "nlp_cfg_template",
        "config_seq": ["hs_index_data", "hs_setup"]
    }
}

test_setup_dicts = {
    'text_sum': {"data": "data_d1", "nlp_cfg": 'text_sum_nlp_cfg'},
    'spacy_classifier': {"data": "data_d2", "nlp_cfg": 'spacy_classifier_nlp_cfg'},
    'hs_search': {"data": "data_d5", "nlp_cfg": 'hs_search_nlp_cfg'},
    't5_base': {"data": "data_d6", "nlp_cfg": 't5_base_nlp_cfg'},
    't5_pipeline': {"data": "data_d6", "nlp_cfg": 't5_pipe_nlp_cfg'},
    'text_embed': {"data": "data_d4", "nlp_cfg": 'text_embed_nlp_cfg'},
    'alt_utt': {"data": "data_d3", "nlp_cfg": 'alt_utt_nlp_cfg'}
}


def eapl_nlp_regression_test(data={}, cfg={}):
    func = nlp_func_map['eapl_data_process_fk_flow']
    data_dicts = {}
    for test_name, test_cfgs in test_setup_dicts.items():
        data = data_templates_dict[test_cfgs['data']]
        nlp_cfg = reg_test_nlp_cfg[test_cfgs['nlp_cfg']]

        try:
            data = func(data, nlp_cfg)
            data_dicts.update({test_name: data})
            nlp_regression_test.debug(f"{test_name} is successful")

        except Exception as e:
            nlp_regression_test.debug(f"{test_name} is failed\n{e}")

    return data_dicts


regression_test_fmap = {
    "eapl_nlp_regression_test": eapl_nlp_regression_test
}
nlp_func_map.update(regression_test_fmap)

if __name__ == '__main__':
    # Logging Support
    logging.basicConfig(
        level='DEBUG',
        format="%(asctime)s %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler('nlp_pipeline.log', mode='w'),
            logging.StreamHandler()
        ])

    eapl_nlp_regression_test()

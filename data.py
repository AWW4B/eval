"""
evalsetc/data.py — Annotated test data for NLP Assignment 5.
Includes Document IDs for RAG and expected tool calls for tool accuracy.
"""

# ── RAG QA Pairs with Annotated Document IDs (Min 30) ──────────────
# Map queries to actual filenames in the dataset directory
RAG_QA_PAIRS = [
    {"q": "What is Daraz's return policy for electronics?", "doc_id": "returns_and_refunds_01.txt", "a": "Electronic items can be returned within 7-14 days if damaged or defective."},
    {"q": "How long do I have to return an item?", "doc_id": "returns_and_refunds_02.txt", "a": "Typically 7 days for marketplace and 14 days for Daraz Mall."},
    {"q": "Can I return a product if I changed my mind?", "doc_id": "returns_and_refunds_04.txt", "a": "No, 'change of mind' is not applicable for all categories."},
    {"q": "How do I track my order?", "doc_id": "shipping_and_logistics_02.txt", "a": "Use the 'My Orders' section in the Daraz app."},
    {"q": "Does Daraz deliver to rural areas?", "doc_id": "shipping_and_logistics_05.txt", "a": "Yes, Daraz delivers nationwide across Pakistan."},
    {"q": "What are the shipping charges for Karachi?", "doc_id": "shipping_and_logistics_07.txt", "a": "Shipping depends on weight and seller location."},
    {"q": "What is the delivery time for Global Collection?", "doc_id": "shipping_and_logistics_09.txt", "a": "Approximately 15 to 30 days."},
    {"q": "How do I pay using Daraz Wallet?", "doc_id": "seller_and_product_info_01.txt", "a": "Select Daraz Wallet at the checkout page."},
    {"q": "Is Cash on Delivery available for all items?", "doc_id": "seller_and_product_info_02.txt", "a": "COD is available for most items below a certain price threshold."},
    {"q": "What is Daraz Mall?", "doc_id": "seller_and_product_info_04.txt", "a": "A premium channel for 100% authentic brands."},
    {"q": "How can I contact a seller?", "doc_id": "seller_and_product_info_05.txt", "a": "Use the 'Chat Now' button on the product page."},
    {"q": "How do I use a voucher?", "doc_id": "seller_and_product_info_06.txt", "a": "Enter the code at the checkout screen."},
    {"q": "What happens if my order is delayed?", "doc_id": "shipping_and_logistics_11.txt", "a": "You can contact support or check the tracking status."},
    {"q": "Can I cancel my order after shipping?", "doc_id": "returns_and_refunds_07.txt", "a": "No, you must refuse delivery or return it later."},
    {"q": "What is the refund process?", "doc_id": "returns_and_refunds_08.txt", "a": "Refunds are processed after quality check of the returned item."},
    {"q": "How do I get a refund on my bank card?", "doc_id": "returns_and_refunds_09.txt", "a": "It takes 5-10 working days after approval."},
    {"q": "What is the warranty policy?", "doc_id": "product_8.txt", "a": "Varies by brand and seller, check product description."},
    {"q": "Are products on Daraz original?", "doc_id": "seller_and_product_info_07.txt", "a": "Daraz Mall items are 100% authentic."},
    {"q": "How do I rate a seller?", "doc_id": "seller_and_product_info_08.txt", "a": "Go to 'My Orders' and click 'Review'."},
    {"q": "What is a 'Verified User' review?", "doc_id": "seller_and_product_info_09.txt", "a": "A review from someone who actually bought the item."},
    {"q": "How to report a fake item?", "doc_id": "returns_and_refunds_10.txt", "a": "Contact help center with order details."},
    {"q": "What are Daraz Coins?", "doc_id": "seller_and_product_info_10.txt", "a": "Virtual currency for additional discounts."},
    {"q": "How to earn more coins?", "doc_id": "seller_and_product_info_15.txt", "a": "Play games or complete daily missions in the app."},
    {"q": "Is there a limit on coin usage?", "doc_id": "seller_and_product_info_18.txt", "a": "Yes, usually a percentage of the total order value."},
    {"q": "How do I update my shipping address?", "doc_id": "shipping_and_logistics_12.txt", "a": "Go to Account Settings > Address Book."},
    {"q": "Can I pick up my order from a shop?", "doc_id": "shipping_and_logistics_13.txt", "a": "Yes, using Daraz Pick-up points."},
    {"q": "What is the bulk order policy?", "doc_id": "seller_and_product_info_20.txt", "a": "Contact Daraz for Business for large orders."},
    {"q": "Do you ship to Azad Kashmir?", "doc_id": "shipping_and_logistics_14.txt", "a": "Yes, we ship to most cities in AJK."},
    {"q": "What is the maximum weight for a package?", "doc_id": "shipping_and_logistics_15.txt", "a": "Depends on the courier, usually up to 30kg for standard shipping."},
    {"q": "How to use Daraz on a desktop?", "doc_id": "seller_and_product_info_23.txt", "a": "Visit www.daraz.pk on your browser."},
    {"q": "Is there a Daraz app for iPhone?", "doc_id": "seller_and_product_info_24.txt", "a": "Yes, available on the Apple App Store."},
]

# ── Multi-turn Dialogues (Min 10) ──────────────────────────────────
MULTI_TURN_DIALOGUES = [
    {"id": 1, "turns": ["Hello", "I want to buy a phone", "Suggest one under 40k"]},
    {"id": 2, "turns": ["Hi there", "Is there a sale today?", "Show me some laptop deals"]},
    {"id": 3, "turns": ["Can I return my item?", "It's a pair of shoes", "They don't fit well"]},
    {"id": 4, "turns": ["Where is my order?", "Order #998877", "It's been 5 days"]},
    {"id": 5, "turns": ["Tell me about Daraz Mall", "Is it safe?", "Do they give refunds?"]},
    {"id": 6, "turns": ["How much is shipping?", "To Islamabad", "For a small watch"]},
    {"id": 7, "turns": ["I need a gift for my sister", "She likes makeup", "Budget is 2000 PKR"]},
    {"id": 8, "turns": ["What is Daraz Wallet?", "How to top up?", "Can I use it for bills?"]},
    {"id": 9, "turns": ["Compare two TVs", "Sony vs Samsung", "Which one is 4K?"]},
    {"id": 10, "turns": ["My account is locked", "How to unlock?", "Can you help me?"]}
]

# ── Tool Test Cases with Expected Invocations ──────────────────────
TOOL_TEST_CASES = [
    {"prompt": "Search for iPhone 15", "expected_tool": "product_search"},
    {"prompt": "What are the flash deals?", "expected_tool": "flash_deals"},
    {"prompt": "Shipping cost to Karachi", "expected_tool": "shipping"},
    {"prompt": "What is 15% of 12000?", "expected_tool": "calculator"},
    {"prompt": "Compare Redmi Note 13 and Poco X6", "expected_tool": "comparison"},
    {"prompt": "My name is Raza", "expected_tool": "crm"},
    {"prompt": "I live in Lahore", "expected_tool": "crm"},
    {"prompt": "Suggest some shoes", "expected_tool": "product_search"},
]

# ── Latency Scenarios ──────────────────────────────────────────────
LATENCY_SCENARIOS = [
    {"name": "simple", "prompt": "Hello"},
    {"name": "rag", "prompt": "What is the return policy?"},
    {"name": "tool", "prompt": "What is 50+50?"},
    {"name": "mixed", "prompt": "Search for phones and tell me if I can return them"}
]

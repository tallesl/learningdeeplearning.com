---
title: "A Heretical Theory No Longer"
date: 2024-06-16
---

When talking about Artificial Intelligence, Turing's name mostly comes due to its **"Turing Test"**, which is
[easily dismissed](https://en.wikipedia.org/wiki/Turing_test#Weaknesses) or even made fun of:

![](/images/a-heretical-theory-no-longer/comic.png)

But Turing thoughts on thinking machines are not as superficial as it may seem at first glance.

## "something very close to thinking"

Back in 1951, Alan Turing gave a presentation arguing against the claim that **"you cannot make a machine think for
you"**. Titled as **"A Heretical Theory"**, it's interesting to see how things changed, what was heretical some decades
ago is now what's in vogue.

Even though he is discussing in layman's terms in the article, we can pinpoint passages that are akin to modern machine
learning techniques.

> "They will make mistakes at times, and at times they may make new and very interesting statements, and on the whole
the output of them will be worth attention to the same sort of extent as the output of a human mind." 

We can relate this passage to Generative AI as a whole (generating new data), and to [AI hallucination](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence))
(or should I say [confabulation](https://en.wikipedia.org/wiki/Confabulation)?).

> "If the machine were able in some way to 'learn by experience' it would be much more impressive. (...) This process
could probably be hastened by a suitable selection of the experiences to which it was subjected. This might be called
'education'." 

[Supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) fits well here.

> "Another problem raised by this picture of the way behaviour is determined is the idea of 'favourable outcome'.
Without some such idea, corresponding to the 'pleasure principle' of the psychologists, it is very difficult to see how
to proceed." 

Sounds like a [loss function](https://en.wikipedia.org/wiki/Loss_functions_for_classification) to me. It's fascinating
to see the push towards how the human mind works bringing [a Freudian term](https://en.wikipedia.org/wiki/Pleasure_principle_(psychology))
to the conversation.

> "There is, however, one feature that I would like to suggest should be incorporated in the machines, and that is a
'random element'. Each machine should be supplied with a tape bearing a random series of figures, e.g., 0 and 1 in
equal quantities, and this series of figures should be used in the choices made by the machine." 

This description is akin to [random weight initialization](https://www.youtube.com/watch?v=6by6Xas_Kho).

> "(...) they would be able to converse with each other to sharpen their wits." 

For this one, I'll let the [2018 Turing Award](https://en.wikipedia.org/wiki/Geoffrey_Hinton) winner answer it for us:

> "We can get information from one to another rather inefficiently by I produce sentences and you figure out how to
change your weights so you would have said the same thing. That's called distillation. But that's a very inefficient
way of communicating knowledge. (...) So digital systems can share weights, and that's incredibly much more efficient.
(...) They all know what all the others learned. We can't do that, and so they're far superior to us in being able to
share knowledge." 

## Sources

- [Existential Comics #15](https://www.existentialcomics.com/comic/15)
- [The Essential Turing](https://academic.oup.com/book/42030)
- [Intelligent Machinery, A Heretical Theory](https://rauterberg.employee.id.tue.nl/lecturenotes/DDM110%20CAS/Turing/Turing-1951%20Intelligent%20Machinery-a%20Heretical%20Theory.pdf)
- [Geoffrey Hinton Interview, minute 24:10](https://www.youtube.com/watch?v=n4IQOBka8bc&t=24m10s)

Since it's not readily available on the web, below is a reproduction of the full article by Turing.

## Intelligent Machinery, A Heretical Theory

A. M. Turing

"You cannot make a machine to think for you." This is a commonplace that is usually accepted without question. It will be the purpose of this paper to question it.

Most machinery developed for commercial purposes is intended to carry out some very specific job, and to carry it out with certainty and considerable speed. Very often it does the same series of operations over and over again without any variety. This fact about the actual machinery available is a powerful argument to many in favour of the slogan quoted above. To a mathematical logician, this argument is not available, for it has been shown that there are machines theoretically possible which will do something very close to thinking. They will, for instance, test the validity of a formal proof in the system of Principia Mathematica, or even tell of a formula of that system whether it is provable or disprovable. In the case that the formula is neither provable nor disprovable such a machine certainly does not behave in a very satisfactory manner, for it continues to work indefinitely without producing any result at all, but this cannot be regarded as very different from the reaction of the mathematicians, who have for instance worked for hundreds of years on the question as to whether Fermat's last theorem is true or not. For the case of machines of this kind a more subtle kind of argument is necessary. By Gödel's famous theorem, or some similar argument, one can show that however the machine is constructed there are bound to be cases where the machine fails to give an answer, but a mathematician would be able to. On the other hand, the machine has certain advantages over the mathematician. Whatever it does can be relied upon, assuming no mechanical 'breakdown', whereas the mathematician makes a certain proportion of mistakes. I believe that this danger of the mathematician making mistakes is an unavoidable corollary of his power of sometimes hitting upon an entirely new method. This seems to be confirmed by the well-known fact that the most reliable people will not usually hit upon really new methods.

My contention is that machines can be constructed which will simulate the behaviour of the human mind very closely. They will make mistakes at times, and at times they may make new and very interesting statements, and on the whole the output of them will be worth attention to the same sort of extent as the output of a human mind. The content of this statement lies in the greater frequency expected for the true statements, and it cannot, I think, be given an exact statement. It would not, for instance, be sufficient to say simply that the machine will make any true statement sooner or later, for an example of such a machine would be one which makes all possible statements sooner or later. We know how to construct these, and as they would (probably) produce true and false statements about equally frequently, their verdicts would be quite worthless. It would be the actual reaction of the machine to circumstances that would prove my contention, if indeed it can be proved at all.

Let us go rather more carefully into the nature of this 'proof'. It is clearly possible to produce a machine which would give a very good account of itself for any range of tests if the machine were made sufficiently elaborate. However, this again would hardly be considered an adequate proof. Such a machine would give itself away by making the same sort of mistake over and over again, and being quite unable to correct itself, or to be corrected by argument from outside. If the machine were able in some way to 'learn by experience' it would be much more impressive. If this were the case there seems to be no real reason why one should not start from a comparatively simple machine, and, by subjecting it to a suitable range of 'experience', transform it into one which was much more elaborate, and was able to deal with a far greater range of contingencies. This process could probably be hastened by a suitable selection of the experiences to which it was subjected. This might be called 'education'. But here we have to be careful. It would be quite easy to arrange the experiences in such a way that they automatically caused the structure of the machine to build up into a previously intended form, and this would obviously be a gross form of cheating, almost on a par with having a man inside the machine. Here again the criterion as to what would be considered reasonable in the way of 'education' cannot be put into mathematical terms, but I suggest that the following would be adequate in practice. Let us suppose that it is intended that the machine shall understand English, and that owing to its having no hands or feet, and not needing to eat, not desiring to smoke, it will occupy its time mostly in playing games such as Chess and GO, and possibly Bridge. The machine is provided with a typewriter keyboard on which any remarks to it are typed, and it also types out any remarks that it wishes to make. I suggest that the education of the machine should be entrusted to some highly competent schoolmaster who is interested in the project but who is forbidden any detailed knowledge of the inner workings of the machine. The mechanic who has constructed the machine, however, is permitted to keep the machine in running order, and if he suspects that the machine has been operating incorrectly may put it back to one of its previous positions and ask the schoolmaster to repeat his lessons from that point on, but he may not take any part in the teaching. Since this procedure would only serve to test the bona fides of the mechanic, I need hardly say that it would not be adopted in the experimental stages. As I see it, this education process would in practice be essential to the production of a reasonably intelligent machine within a reasonably short space of time. The human analogy alone suggests this.

I may now give some indication of the way in which such a machine might be expected to function. The machine would incorporate a memory. This does not need very much explanation. It would simply be a list of all the statements that had been made to it or by it, and all the moves it had made and the cards it had played in its games. These would be listed in chronological order. Besides this straightforward memory there would be a number of 'indexes of experiences'. To explain this idea I will suggest the form which one such index might possibly take. It might be an alphabetical index of the words that had been used giving the 'times' at which they had been used, so that they could be looked up in the memory. Another such index might contain patterns of men or parts of a GO board that had occurred. At comparatively late stages of education the memory might be extended to include important parts of the configuration of the machine at each moment, or in other words it would begin to remember what its thoughts had been. This would give rise to fruitful new forms of indexing. New forms of index might be introduced on account of special features observed in the indexes already used. The indexes would be used in this sort of way. Whenever a choice has to be made as to what to do next features of the present situation are looked up in the indexes available, and the previous choice in the similar situations, and the outcome, good or bad, is discovered. The new choice is made accordingly. This raises a number of problems. If some of the indications are favourable and some are unfavourable what is one to do? The answer to this will probably differ from machine to machine and will also vary with its degree of education. At first probably some quite crude rule will suffice, e.g., to do whichever has the greatest number of votes in its favour. At a very late stage of education the whole question of procedure in such cases will probably have been investigated by the machine itself, by means of some kind of index, and this may result in some highly sophisticated, and, one hopes, highly satisfactory, form of rule. It seems probable, however, that the comparatively crude forms of rule will themselves be reasonably satisfactory, so that progress can on the whole be made in spite of the crudeness of the choice rules. This seems to be verified by the fact that Engineering problems are sometimes solved by the crudest rule of thumb procedure which only deals with the most superficial aspects of the problem, e.g., whether a function increases or decreases with one of its variables. Another problem raised by this picture of the way behaviour is determined is the idea of 'favourable outcome'. Without some such idea, corresponding to the 'pleasure principle' of the psychologists, it is very difficult to see how to proceed. Certainly it would be most natural to introduce some such thing into the machine. I suggest that there should be two keys which can be manipulated by the schoolmaster, and which represent the ideas of pleasure and pain. At later stages in education the machine would recognise certain other conditions as desirable owing to their having been constantly associated in the past with pleasure, and likewise certain others as undesirable. Certain expressions of anger on the part of the schoolmaster might, for instance, be recognised as so ominous that they could never be overlooked, so that the schoolmaster would find that it became unnecessary to 'apply the cane' any more.

To make further suggestions along these lines would perhaps be unfruitful at this stage, as they are likely to consist of nothing more than an analysis of actual methods of education applied to human children. There is, however, one feature that I would like to suggest should be incorporated in the machines, and that is a 'random element'. Each machine should be supplied with a tape bearing a random series of figures, e.g., 0 and 1 in equal quantities, and this series of figures should be used in the choices made by the machine. This would result in the behaviour of the machine not being by any means completely determined by the experiences to which it was subjected, and would have some valuable uses when one was experimenting with it. By faking the choices made one would be able to control the development of the machine to some extent. One might, for instance, insist on the choice made being a particular one at, say, 10 particular places, and this would mean that about one machine in 1024 or more would develop to as high a degree as the one which had been faked. This cannot very well be given an accurate statement because of the subjective nature of the idea of 'degree of development' to say nothing of the fact that the machine that had been faked might have been also fortunate in its unfaked choices.

Let us now assume, for the sake of argument, that these machines are a genuine possibility, and look at the consequences of constructing them. To do so would of course meet with great opposition, unless we have advanced greatly in religious toleration from the days of Galileo. There would be great opposition from the intellectuals who were afraid of being put out of a job. It is probable though that the intellectuals would be mistaken about this. There would be plenty to do in trying, say, to keep one's intelligence up to the standard set by the machines, for it seems probable that once the machine thinking method had started, it would not take long to outstrip our feeble powers. There would be no question of the machines dying, and they would be able to converse with each other to sharpen their wits. At some stage therefore we should have to expect the machines to take control, in the way that is mentioned in Samuel Butler's Erewhon. 
